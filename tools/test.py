import argparse
import time
import gorilla
import torch
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from mydesign.dataset import build_dataloader, build_dataset
from mydesign.evaluation import ScanNetEval
from mydesign.model import SPFormer
from mydesign.utils import get_root_logger, save_gt_instances, save_pred_instances


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--config', type=str, help='path to config file', default='configs/spf_scannet.yaml')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument('--num_warmup', type=int, default=5, help='number of warmup iterations')
    parser.add_argument('--num_test', type=int, default=20, help='number of test iterations')
    args = parser.parse_args()
    return args


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_params_count(num_params):
    """格式化参数量显示"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f} B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f} M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f} K"
    else:
        return str(num_params)


def measure_inference_time(model, dataloader, num_warmup=5, num_test=20):
    """测量推理时间"""
    # 预热
    print(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_warmup:
                break
            _ = model(batch, mode='predict')
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # 测试推理时间
    print(f"Testing inference time for {num_test} iterations...")
    times = []
    
    # 重置dataloader
    dataloader_iter = iter(dataloader)
    
    with torch.no_grad():
        for i in range(num_test):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(batch, mode='predict')
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    # 计算统计信息
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    max_time = times.max().item()
    
    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'fps': 1.0 / mean_time
    }


def measure_memory_usage(model, dataloader, num_iterations=5):
    """测量显存占用（单位：MB）"""
    memory_stats = []
    
    # 重置dataloader
    dataloader_iter = iter(dataloader)
    
    with torch.no_grad():
        for i in range(num_iterations):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # 清空缓存
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # 运行推理
            _ = model(batch, mode='predict')
            
            if torch.cuda.is_available():
                # 获取当前显存使用（转换为MB）
                current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                max_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                cached_memory = torch.cuda.memory_reserved() / 1024**2  # MB
                
                memory_stats.append({
                    'current': current_memory,
                    'peak': max_memory,
                    'cached': cached_memory
                })
    
    # 计算平均显存使用
    if memory_stats:
        avg_current = sum(m['current'] for m in memory_stats) / len(memory_stats)
        avg_peak = sum(m['peak'] for m in memory_stats) / len(memory_stats)
        avg_cached = sum(m['cached'] for m in memory_stats) / len(memory_stats)
        
        return {
            'current': avg_current,
            'peak': avg_peak,
            'cached': avg_cached
        }
    return None

def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    # 加载模型
    model = SPFormer(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\n{'='*50}")
    logger.info("Model Parameters Summary:")
    logger.info(f"Total parameters: {format_params_count(total_params)}")
    logger.info(f"Trainable parameters: {format_params_count(trainable_params)}")
    logger.info(f"Non-trainable parameters: {format_params_count(total_params - trainable_params)}")
    logger.info(f"{'='*50}\n")

    # 准备数据
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)
    
    # 设置模型为评估模式
    model.eval()
    
    # 测量推理时间
    logger.info("Measuring inference time...")
    time_stats = measure_inference_time(model, dataloader, args.num_warmup, args.num_test)
    logger.info(f"\n{'='*50}")
    logger.info("Inference Time Statistics:")
    logger.info(f"Mean inference time: {time_stats['mean']*1000:.2f} ms")
    logger.info(f"Std inference time: {time_stats['std']*1000:.2f} ms")
    logger.info(f"Min inference time: {time_stats['min']*1000:.2f} ms")
    logger.info(f"Max inference time: {time_stats['max']*1000:.2f} ms")
    logger.info(f"FPS: {time_stats['fps']:.2f}")
    logger.info(f"{'='*50}\n")
    
    # 测量显存占用
    if torch.cuda.is_available():
        logger.info("Measuring GPU memory usage...")
        memory_stats = measure_memory_usage(model, dataloader, num_iterations=5)
        if memory_stats:
            logger.info(f"\n{'='*50}")
            logger.info("GPU Memory Usage Statistics:")
            logger.info(f"Average current memory: {memory_stats['current']:.2f} MB")
            logger.info(f"Average peak memory: {memory_stats['peak']:.2f} MB")
            logger.info(f"Average cached memory: {memory_stats['cached']:.2f} MB")
            logger.info(f"{'='*50}\n")
    
    # 继续原有的评估流程
    logger.info("Starting evaluation...")
    results, scan_ids, pred_insts, gt_insts = [], [], [], []

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            result = model(batch, mode='predict')
            results.append(result)
            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not cfg.data.test.prefix == 'test':
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(pred_insts, gt_insts)

    # save output
    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)
    
    logger.info("All done!")


if __name__ == '__main__':
    main()