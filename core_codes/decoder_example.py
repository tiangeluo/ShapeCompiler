    class PointVQVAE_Decoder(nn.Module):
        def __init__(self, feat_dims, codebook_dim=512, final_dim=2048):
            super(PointVQVAE_Decoder, self).__init__()
            self.dim = codebook_dim
            self.folding1 = nn.Sequential(
                nn.Conv1d(self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, self.dim, 1),
            )
            
            self.folding2 = nn.Sequential(
                nn.Conv1d(self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, self.dim, 1),
            )
            self.folding3 = nn.Sequential(
                nn.Conv1d(self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, 4*self.dim, 1),
                nn.BatchNorm1d(4*self.dim),
                nn.ReLU(),
                nn.Conv1d(4*self.dim, self.dim, 1),
            )
            self.our_end = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(self.dim, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Conv1d(1024, final_dim*3, 1)
            )

        def forward(self, x):
            # x.shape = (batch_size, codebook_dim: 512, part_embeding_num: 128)
            folding_result1 = self.folding1(x)              
            # folding_result1.shape = (batch_size, codebook_dim: 512, part_embeding_num: 128)
            x = x+folding_result1
            folding_result2 = self.folding2(x)              
            # folding_result2.shape = (batch_size, codebook_dim: 512, part_embeding_num: 128)
            x = x+folding_result2
            folding_result3 = self.folding3(x)              
            # folding_result3.shape = (batch_size, codebook_dim: 512, part_embeding_num: 128)
            x = x+folding_result3
            max_feature = torch.max(x, -1, keepdim=True)[0] 
            # max_feature.shape = (batch_size, codebook_dim: 512, 1)
            output = self.our_end(max_feature)              
            # output.shape = (batch_size, 3*2048, 1)
            return output          