import torch
import torch.nn as nn
import gmic
import cv2
import numpy as np
import read_config_file
from torchvision import transforms

device = 'cuda'

class SILmodel(nn.Module):
    def __init__(self, config_params):
        super(SILmodel, self).__init__()
        self.activation = config_params['activation']
        self.featureextractormodel = config_params['femodel']
        self.extra = config_params['extra']
        self.topkpatch = config_params['topkpatch']
        self.pretrained = config_params['pretrained']
        self.channel = config_params['channel']
        self.regionpooling = config_params['regionpooling']
        self.learningtype = config_params['learningtype']
        
        if self.featureextractormodel:
            self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])
    
    def forward(self, x, eval_mode=True):
        return self.feature_extractor(x, eval_mode)
        y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec = self.feature_extractor(x, eval_mode)
        return y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec


class GMICModel(nn.Module):
    def __init__(self, config_params, model_path):
        super(GMICModel, self).__init__()
        self.model = SILmodel(config_params)
        self.transforms = self.preprocess()
        self.load_model(model_path)
    
    def forward(self, images):
        images = self.transforms(images).to('cuda')
        # import pudb; pudb.set_trace()
        return self.model(images)
        y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec = self.model(images)
        return concat_vec

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        print("checkpoint epoch and loss:", checkpoint['epoch'], checkpoint['loss'])  
        self.model.load_state_dict(checkpoint['state_dict'])

    def preprocess(self):
        preprocess_test = transforms.Compose([transforms.Resize((2944,1920))])
        return preprocess_test


def build_model(device):
    config_file = '/lustre/pathak/runs/embed/train_gmic_embed/config_80_8.ini'    
    config_params = read_config_file.read_config_file(config_file)
    model_path = '/lustre/pathak/runs/embed/train_gmic_embed/model_80_8.tar' 
    model = GMICModel(config_params, model_path)
    print('model loaded')
    model.to(device)
    return model

def load_image(img_path):
    img = cv2.imread(img_path,-1)
    img_dtype = img.dtype
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
    if img_dtype=='uint16':
        img/=65535
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    return img


if __name__=='__main__':
    #read arguments
    config_file = '/lustre/pathak/runs/embed/train_gmic_embed/config_80_8.ini'    
    config_params = read_config_file.read_config_file(config_file)

    path_to_model = '/lustre/pathak/runs/embed/train_gmic_embed/model_80_8.tar' 
        
    model = build_model(config_params, path_to_model, device)
    model.eval()

    img_path = '/lustre/pathak/data/embed/processed_images/1003266064259604/LCC_png_images-cohort_1-842ee092b0e93ac76ed7ce56f797a023d4bdbdda58b4e6bfe36d08f2-9ab611e24fb7f13a3996adf5705c57194db4124cafa5eb485df6a256-a00cc4664f2ddbce5d7623ac4c16159f7ab5aef768b56fb13b8dd140.png'
    img = load_image(img_path)
    print(img.shape)
    pred = model(img)
    print(pred.shape)
    # """Pathak Testing code"""
    # total_images=0
    # correct = 0
    # s=0
    # batch_test_no=0
    # count_dic_viewwise={model_pat
    # eval_subgroup = False
    # eval_mode = True
    # conf_mat_test=np.zeros((config_params['numclasses'],config_params['numclasses']))
    # views_standard=['LCC', 'LMLO', 'RCC', 'RMLO']

    # with torch.no_grad():
    #     for test_idx, test_batch, test_labels, views_names in dataloader_test:
    #         test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
    #         test_labels = test_labels.view(-1)

            
    #         output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(test_batch) # compute model output, loss and total train loss over one epoch
    #         output_patch_test = None
            
    #         output_batch_local = output_batch_local.view(-1)
    #         output_batch_global = output_batch_global.view(-1)
    #         output_batch_fusion = output_batch_fusion.view(-1)
    #         test_labels = test_labels.float()
    #         test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
    #         output_test = output_batch_fusion
            
            
    #         if batch_test_no==0:
    #             test_pred_all=test_pred
    #             test_labels_all=test_labels
    #             print(output_test.data.shape, flush=True)
    #             output_all_ten=torch.sigmoid(output_test.data)
    #         else:
    #             test_pred_all=torch.cat((test_pred_all,test_pred),dim=0)
    #             test_labels_all=torch.cat((test_labels_all,test_labels),dim=0)
    #             output_all_ten=torch.cat((output_all_ten,torch.sigmoid(output_test.data)),dim=0)
                
                
    #         correct, total_images, conf_mat_test, conf_mat_batch = evaluation.conf_mat_create(test_pred, test_labels, correct, total_images, conf_mat_test, config_params['classes'])
    #         batch_test_no+=1
    #         s=s+test_labels.shape[0]
    #         print('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1), flush=True)
