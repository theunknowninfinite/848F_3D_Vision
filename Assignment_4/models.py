import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3,N=10000):
        super(cls_model, self).__init__()


        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_maxpool = nn.MaxPool1d(N)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout=nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Permute the input to (B, 3, N)
        x = points.permute(0, 2, 1).cuda()

        # MLP 64-128-1024
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = self.global_maxpool(x)

        # MLP for classification
        x = F.relu(self.fc1(x.squeeze()))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6,N=10000):
        super(seg_model, self).__init__()
        #MLP 64-64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        #MLP 64-1024
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)

        self.global_maxpool = nn.MaxPool1d(N)
        #MLP for Segmentation
        self.conv5 = nn.Conv1d(1088, 512, 1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.bn6 = nn.BatchNorm1d(256)
        self.conv7 = nn.Conv1d(256, 128, 1)
        self.bn7 = nn.BatchNorm1d(128)
        #final conv
        self.conv8 = nn.Conv1d(128, num_seg_classes, 1)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        N=points.shape[1]
        # Permute the input to (B, 3, N)
        x = points.permute(0, 2, 1).cuda()
        #MLP 64-64
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        local_feat=x
        #MLP 64-1024
        x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x)))

        #Global max pooling
        # x=torch.max(x,dim=-1)
        x=self.global_maxpool(x).squeeze() # Bx1088
        global_feat=x
        # print(global_feat.shape,local_feat.shape)
        global_feat=global_feat.repeat(1,N).view(-1,N,global_feat.shape[1])
        local_feat=local_feat.permute(0,2,1)

        #Concatenate global features with local features
        # print(global_feat.shape,local_feat.shape)
        x=torch.cat((local_feat,global_feat),dim=-1).permute(0,2,1)
        # print(x.shape)

        #MLP for Segmentation
        x=F.relu(self.bn5(self.conv5(x)))
        x=F.relu(self.bn6(self.conv6(x)))
        x=F.relu(self.bn7(self.conv7(x)))
        #Final conv
        x=self.conv8(x)
        # print(x.shape)
        x=x.permute(0,2,1)

        return x
