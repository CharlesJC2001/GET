import torch
import torch.nn as nn

# class DyT(nn.Module):
    
#     def __init__(self, num_features, alpha_init_value=0.5):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
#         self.weight = nn.Parameter(torch.ones(num_features))
#         self.bias = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         x = torch.tanh(self.alpha * x)
#         return x * self.weight + self.bias

class ChannelAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # 对特征维度压缩
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 16),  # 压缩通道
            nn.ReLU(),
            nn.Linear(d_model // 16, d_model),  # 恢复通道
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (M, d_model)
        weights = self.fc(torch.mean(x, dim=0, keepdim=True))  # (1, d_model)
        return x * weights  # 广播乘法
    
class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs

class SpatialCrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def get_coords_weight(self, source_coord, query_coord):
        """
        source_coord (N, 3)
        query_coord (n, 3)
        """
        beta = 0.9
        # distance_matrix = (query_coord.unsqueeze(1) - source_coord.unsqueeze(0)).abs().sum(dim=2)
        # distance_matrix = torch.sqrt(((query_coord.unsqueeze(1) - source_coord.unsqueeze(0)) ** 2).sum(dim=2))
        distance_matrix = (query_coord.unsqueeze(1) - source_coord.unsqueeze(0)).abs().max(dim=2)[0]
        coords_weight = torch.pow(beta, distance_matrix)

        # mean_coords = torch.mean(coords_weight)
        # attn_mask = (coords_weight < mean_coords).bool()
        # return coords_weight, attn_mask # (n,N), (n,N)
        return coords_weight # (n,N)

    def forward(self, source, query, source_coords, query_coords, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        source_coords (B*N, 3)
        query_coords (B, n, 3)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            source_ori = source[start_id:end_id]
            source_coord = source_coords[start_id:end_id]
            query_coord = query_coords[i]
            v = source_ori.unsqueeze(0)
            q_ori = query[i]
            k_ori = source_ori
            coords_weight = self.get_coords_weight(source_coord, query_coord)
            # coords_weight, attn_mask_spatial = self.get_coords_weight(source_coord, query_coord)
            # attn_masks[i][~attn_mask_spatial] = False
            # 计算coords_weight的SVD分解
            U, S, Vh = torch.linalg.svd(coords_weight)
            # 取最大奇异值对应的向量
            u_SVD = U[:, 0]  # (N,)
            v_SVD = Vh[0, :]  # (n,)
            s_SVD = S[0]  # 标量
            # 计算q和k
            q = (q_ori * u_SVD.reshape(-1, 1)).unsqueeze(0)  # (1, N, d)
            k = (k_ori * v_SVD.reshape(-1, 1) * s_SVD).unsqueeze(0)  # (1, n, d)
            if attn_masks:
                output, _ = self.attn(q, k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(q, k, v)
            self.dropout(output)
            output = output + q_ori
            self.norm(output)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # (b, 100, d_model)
        return outputs

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


# class SpatialSelfAttentionLayer(nn.Module):

#     def __init__(self, d_model=256, nhead=8, dropout=0.0):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             d_model,
#             nhead,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
    
#     def get_coords_weight(self, query_coord):
#         """
#         query_coord (n, 3)
#         """
#         beta = 0.8
#         distance_matrix = (query_coord.unsqueeze(1) - query_coord.unsqueeze(0)).abs().sum(dim=2)
#         coords_weight = torch.pow(beta, distance_matrix)
#         return coords_weight # (n,n)

#     def forward(self, x, query_coords, pe=None):
#         """
#         x Tensor (b, 100, c)
#         query_coords (B, n, 3)
#         """

#         B = x.shape[0]
#         outputs = []
#         x = self.with_pos_embed(x, pe)
#         for i in range(B):
#             q_ori = x[i]
#             query_coord = query_coords[i]
#             coords_weight = self.get_coords_weight(query_coord)
#             # 计算coords_weight的Cholesky分解
#             L = torch.linalg.cholesky(coords_weight)
#             # 计算q和k
#             q = k = q_ori * L.unsqueeze(1)  # (n, d)
#             output, _ = self.attn(q, k, q_ori)
#             output = self.dropout(output) + q_ori
#             output = self.norm(output)
#             outputs.append(output)
#         outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
#         return outputs

class SpatialSelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_coords_weight(self, query_coord):
        """
        query_coord (n, 3)
        """
        beta = 0.9
        # distance_matrix = (query_coord.unsqueeze(1) - query_coord.unsqueeze(0)).abs().sum(dim=2)
        # distance_matrix = torch.sqrt(((query_coord.unsqueeze(1) - query_coord.unsqueeze(0)) ** 2).sum(dim=2))
        distance_matrix = (query_coord.unsqueeze(1) - query_coord.unsqueeze(0)).abs().max(dim=2)[0]
        coords_weight = torch.pow(beta, distance_matrix)
        mean_coords = torch.mean(coords_weight)
        attn_mask = (coords_weight < mean_coords).bool()
        return coords_weight, attn_mask # (n,n),(n,n)

    def forward(self, x, query_coords, pe=None):
        """
        x Tensor (b, 100, c)
        query_coords (B, n, 3)
        """

        B = x.shape[0]
        outputs = []
        x = self.with_pos_embed(x, pe)
        for i in range(B):
            q_ori = x[i]
            query_coord = query_coords[i]
            coords_weight, attn_mask = self.get_coords_weight(query_coord)
            q = k = q_ori # (n, d)
            output, _ = self.attn(q, k, q_ori, attn_mask=attn_mask)
            output = self.dropout(output) + q_ori
            output = self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b*100, d_model)
        return outputs


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class QueryDecoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        pe=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.get_query = FFN(d_model, hidden_dim, dropout, activation_fn)
        # self.get_query = FFN(d_model, d_model/2, dropout, activation_fn)
        if pe:
            self.pe = nn.Embedding(num_query, d_model)
        self.spatial_cross_attn_layers = nn.ModuleList([])
        self.spatial_self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.spatial_cross_attn_layers.append(SpatialCrossAttentionLayer(d_model, nhead, dropout))
            self.spatial_self_attn_layers.append(SpatialSelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        # self.inst_channel = ChannelAttention(d_model)
        # self.mask_channel = ChannelAttention(d_model)

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_offsets):
        query = self.out_norm(query)
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_scores, pred_masks, attn_masks

    def get_query_sample(self, inst_feats, x_mars, sp_coords, batch_offsets):
        """
        x_mar [B*M, media]
        """
        query_samples = []
        sample_ids = []
        query_coords = []
        B = len(batch_offsets)-1
        for i in range(B):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            inst_feat = inst_feats[start_id: end_id] # (M, d_model)
            sp_coord = sp_coords[start_id: end_id] # (M, 3)
            M = inst_feat.shape[0]
            d_model = inst_feats.shape[1]
            x_mar = x_mars[start_id: end_id] # (M，media)
            x_score = torch.mean(x_mar, dim=1) # (M)
            if M < self.num_query:
                _, query_sample_id = torch.topk(x_score, k=M, dim=-1, largest=False)
                # query_sample = inst_feat[query_sample_id]
                query_sample = inst_feat[query_sample_id].detach()
                query_zeros = torch.zeros(self.num_query-M, d_model, dtype=query_sample.dtype, device=query_sample.device)
                query_pad = torch.cat((query_sample, query_zeros), dim=0)
                query_samples.append(query_pad)
                query_coord = sp_coord[query_sample_id].detach()
                query_coord_zeros = torch.zeros(self.num_query-M, 3, dtype=query_coord.dtype, device=query_coord.device)
                query_coord_pad = torch.cat((query_coord, query_coord_zeros), dim=0)
                query_coords.append(query_coord_pad)
                sample_ids.append(query_sample_id)
            else:
                _, query_sample_id = torch.topk(x_score, k=self.num_query, dim=-1, largest=False)
                # query_sample = inst_feat[query_sample_id]
                query_sample = inst_feat[query_sample_id].detach()
                query_samples.append(query_sample)
                query_coord = sp_coord[query_sample_id].detach()
                query_coords.append(query_coord)
                sample_ids.append(query_sample_id)
        assert len(query_samples) == B
        query_sample_batched = torch.stack(query_samples, dim=0)# (b, 400, d_model)
        query_coords_batched = torch.stack(query_coords, dim=0)# (b, 400, 3)
        return query_sample_batched, sample_ids, query_coords_batched

    def forward_simple(self, x, x_mar, sp_coords, batch_offsets):
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
        pred_labels, pred_scores, pred_masks, _ = self.prediction_head(query, mask_feats, batch_offsets)
        return {'labels': pred_labels, 'masks': pred_masks, 'scores': pred_scores}

    def forward_iter_pred(self, x, x_mar, sp_coords, batch_offsets):
        """
        x [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        # inst_feats = self.inst_channel(inst_feats)
        # mask_feats = self.mask_channel(mask_feats)
        # mask_feats -= 1e-6
        B = len(batch_offsets) - 1
        d_model = inst_feats.shape[-1]
        if getattr(self, 'pe', None):
            pe = self.pe.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            pe = None
        query_sample, sample_ids, query_coords= self.get_query_sample(inst_feats, x_mar, sp_coords, batch_offsets) # (b, n, d_model), (b, n), (b, n, 3)
        query = self.get_query(query_sample)

        pred_labels, pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_offsets)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        for i in range(self.num_layer):
            query = self.spatial_cross_attn_layers[i](inst_feats, query, sp_coords, query_coords, batch_offsets, attn_masks, pe)
            query = self.spatial_self_attn_layers[i](query,query_coords, pe)
            query = self.ffn_layers[i](query)
            pred_labels, pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_offsets)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        return {
            'labels':
            pred_labels,
            'masks':
            pred_masks,
            'scores':
            pred_scores,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c
            } for a, b, c in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }, mask_feats, sample_ids

    def forward(self, x, x_mar, sp_coords, batch_offsets):
        if self.iter_pred:
            return self.forward_iter_pred(x, x_mar, sp_coords, batch_offsets)
        else:
            return self.forward_simple(x, x_mar, sp_coords, batch_offsets)
