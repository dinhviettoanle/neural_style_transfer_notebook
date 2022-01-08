# Just to make it handier, we define it as a nn.Module (like nn.MSELoss)
class GramMSELoss(nn.Module):
    def forward(self, feature, target):
        out = nn.MSELoss()(gram_matrix(feature), target) 
        return out