# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMGatedAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, att_type='general', dropout=0.5):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        

        self.dropouta = nn.Dropout(dropout)
        self.dropoutv = nn.Dropout(dropout)
        self.dropoutl = nn.Dropout(dropout)
        
        if att_type == 'av_bg_fusion':
            # Audio-text fusion
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            # Visual-text fusion
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == 'general':
            # Unimodal transforms
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            
            # Bimodal gates
            self.transform_av = nn.Linear(mem_dim*3, 1)
            self.transform_al = nn.Linear(mem_dim*3, 1)
            self.transform_vl = nn.Linear(mem_dim*3, 1)
    
    def forward(self, a, v, l, modals=None):
        """
        Forward pass.
        Args:
            a: audio features [batch_size, seq_len, mem_dim]
            v: visual features [batch_size, seq_len, mem_dim]
            l: text features [batch_size, seq_len, mem_dim]
            modals: active modalities, e.g. ['a', 'v', 'l']
        Returns:
            Fused features.
        """
        if modals is None:
            modals = ['a', 'v', 'l']
        
        # Apply dropout
        a = self.dropouta(a) if len(a) != 0 and 'a' in modals else a
        v = self.dropoutv(v) if len(v) != 0 and 'v' in modals else v
        l = self.dropoutl(l) if len(l) != 0 and 'l' in modals else l
        
        if self.att_type == 'av_bg_fusion':
            outputs = []
            
            # Audio-text fusion
            if 'a' in modals and 'l' in modals:
                fal = torch.cat([a, l], dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa * (self.scalar_al(a))
                outputs.append(hma)
            
            # Visual-text fusion
            if 'v' in modals and 'l' in modals:
                fvl = torch.cat([v, l], dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv * (self.scalar_vl(v))
                outputs.append(hmv)
            
            # Add text features
            if 'l' in modals:
                outputs.append(l)
            
            # Concatenate outputs
            if len(outputs) > 1:
                hmf = torch.cat(outputs, dim=-1)
            else:
                hmf = outputs[0] if outputs else l
            
            return hmf
            
        elif self.att_type == 'general':
            # Unimodal transforms
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l
            
            outputs = []
            
            # Audio-visual fusion
            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a*v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 'l' not in modals:
                    return h_av
                outputs.append(h_av)
            
            # Audio-text fusion
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a*l], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * hl
                if 'v' not in modals:
                    return h_al
                outputs.append(h_al)
            
            # Visual-text fusion
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v*l], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * hl
                if 'a' not in modals:
                    return h_vl
                outputs.append(h_vl)
            
            # Concatenate bimodal outputs
            if len(outputs) > 1:
                return torch.cat(outputs, dim=-1)
            elif len(outputs) == 1:
                return outputs[0]
            else:
                # If no bimodal fusion is used, concatenate unimodal features
                single_outputs = []
                if 'a' in modals:
                    single_outputs.append(ha)
                if 'v' in modals:
                    single_outputs.append(hv)
                if 'l' in modals:
                    single_outputs.append(hl)
                return torch.cat(single_outputs, dim=-1) if single_outputs else hl


class MultiModalFusionLayer(nn.Module):
    """
    Multi-modal fusion layer integrating MMGatedAttention with additional processing.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, att_type='general'):
        super(MultiModalFusionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-modal gated attention
        self.mm_gated_attention = MMGatedAttention(
            mem_dim=input_dim,
            cand_dim=hidden_dim,
            att_type=att_type,
            dropout=dropout
        )
        
        # Determine fused dimension based on attention type
        if att_type == 'general':
            # In general mode, concatenate three bimodal fusion outputs
            fused_dim = hidden_dim * 3
        else:
            # In av_bg_fusion mode
            fused_dim = hidden_dim * 3  # assume three modalities exist
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual projection (if dimensions do not match)
        self.residual_projection = None
        if input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, text_features, audio_features, visual_features, modals=['a', 'v', 'l']):
        """
        Forward pass.
        Args:
            text_features: text features [batch_size, seq_len, input_dim]
            audio_features: audio features [batch_size, seq_len, input_dim]
            visual_features: visual features [batch_size, seq_len, input_dim]
            modals: active modalities
        Returns:
            Fused features [batch_size, seq_len, output_dim].
        """
        # Multi-modal gated attention fusion
        fused_features = self.mm_gated_attention(
            a=audio_features,
            v=visual_features,
            l=text_features,
            modals=modals
        )
        
        # Output projection
        output = self.output_projection(fused_features)
        
        # Residual connection (use text features as residual)
        if self.residual_projection is not None:
            residual = self.residual_projection(text_features)
        else:
            residual = text_features
        
        return output + residual
