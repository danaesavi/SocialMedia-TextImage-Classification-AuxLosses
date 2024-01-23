import sys
sys.path.append("../preprocessing/")
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import normalize
from torchmetrics.classification import (
    F1Score,
    Precision,
    Recall,
    )
import torch
from config import fixed_feat_size, T, TDATA5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_tensor_and_normalize(imagepil): #Done with testing
    """Convert image to torch Tensor and normalize using the ImageNet training
    set mean and stdev taken from
    https://pytorch.org/docs/stable/torchvision/models.html.
    Why the ImageNet mean and stdev instead of the PASCAL VOC mean and stdev?
    Because we are using a model pretrained on ImageNet."""
    input_size = 224
    ChosenTransforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    return ChosenTransforms(imagepil)

def get_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )

def get_conv_layers(model):
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0 
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv}")
    return conv_layers

def get_conv_model(model_name):

    cnn_models = {
        "resnet50":  models.resnet50(),
        "resnet152":  models.resnet152(),
    }
    return cnn_models[model_name]

def masked_mean(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        return torch.mean(input, dim=dim)
    else:
        length = input.size(1)
        mask = mask[:,:length].unsqueeze(-1)
        mask_input = input * mask
        sum_mask_input = mask_input.sum(dim=dim)
        mask_ = mask.sum(dim=dim)
        sum_mask_out = sum_mask_input/mask_
        return sum_mask_out


def masked_max(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        max_v, _ = torch.max(input, dim=dim)
        return max_v
    else:
        length = input.size(1)
        mask = mask[:,:length].unsqueeze(-1)
        mask = mask.repeat(1, 1, input.size(-1))
        input = input.masked_fill(mask == 0.0, float('-inf'))
        max_v, _ = torch.max(input, dim=dim)
        return max_v

def vectorize_labels(y,y_val,y_te,num_labels):
    # train
    y_vector_tr = np.zeros((len(y),num_labels))
    for i,cat in enumerate(list(y)):
        y_vector_tr[i][cat] = 1
    #val
    y_vector_val = np.zeros((len(y_val),num_labels))
    for i,cat in enumerate(list(y_val)):
        y_vector_val[i][cat] = 1
    #test
    y_vector_te = np.zeros((len(y_te),num_labels))
    for i,cat in enumerate(list(y_te)):
        y_vector_te[i][cat] = 1
    return  y_vector_tr, y_vector_val, y_vector_te

def vectorize_labels_random(y,y_val,y_te,num_labels):
    # train
    y_vector_tr = np.random.randint(2, size=(len(y),num_labels), dtype=int)
    #val
    y_vector_val = np.random.randint(2, size=(len(y_val),num_labels), dtype=int)
    #test
    y_vector_te = np.random.randint(2, size=(len(y_te),num_labels), dtype=int)
    return  y_vector_tr, y_vector_val, y_vector_te
    
def prepare_data(data, num_labels, testing=False,nsamples=-1, compute_class_weights=True, random_labels=False,
                 load_image_adds=False, vectorize=True, multilabel=False):
    if testing:
        data = data.sample(200)
    
        print(data.head())
    column_names = ["tweet_id","text","label"] if not load_image_adds else ["tweet_id","text","label","image_adds"]
    # load data
    train = data[data.split=="train"][column_names] 
    if nsamples > 0:
        train = train.sample(nsamples)
    y = train.label.values
    val = data[data.split=="val"][column_names]
    y_val = val.label.values
    test = data[data.split=="test"][column_names] 
    y_te = test.label.values
    if multilabel:
        y = np.asarray([eval(x) for x in y])
        y_val = np.asarray([eval(x) for x in y_val])
        y_te = np.asarray([eval(x) for x in y_te])
        y_vector_tr, y_vector_val, y_vector_te = y,y_val,y_te
    else:
        if vectorize:
            if random_labels:
                y_vector_tr, y_vector_val, y_vector_te = vectorize_labels_random(y,y_val,y_te,num_labels)
            else:
                y_vector_tr, y_vector_val, y_vector_te = vectorize_labels(y,y_val,y_te,num_labels)
        else:
            y_vector_tr, y_vector_val, y_vector_te = np.asarray(y),np.asarray(y_val),np.asarray(y_te)

    

    print("train", len(train))
    print("val", len(val))
    print("test", len(test))
    #print(train.label.value_counts())
    #print("val")
    #print(val.label.value_counts())
    #print("test")
    #print(val.label.value_counts())
    
    if compute_class_weights:
        if not multilabel:
            # class weights
            class_weights = compute_class_weight(
                class_weight="balanced", 
                classes = list(range(num_labels)), 
                y = y)
            print("class_weights", class_weights)   
            class_weights = torch.as_tensor(class_weights, dtype=torch.float).to(device) 
        else:
            count = np.array(y).sum(axis=0)
            print("count", count)
            class_weights = torch.tensor((len(y)-count)/count).to(device)
            print("class_weights", class_weights)
    else:
        class_weights = None

    
    if load_image_adds:
        y_image_adds = {
            "train":train.image_adds.values, 
            "val":val.image_adds.values, 
            "test":test.image_adds.values
        }
    else:
         y_image_adds = {
            "train":None, 
            "val":None, 
            "test":None
        }
    return train, y_vector_tr, val, y_vector_val, test, y_vector_te, class_weights, y_image_adds


def prepare_text_data(num_labels=2, testing=False, load_image_adds=False):
    import pandas as pd
    data_key = pd.read_csv(TDATA5)
    data = data_key[["id","text","label"]].rename(
                columns={"id":"tweet_id"})
    if testing:
        data = data.sample(100)
    
    y_vector = np.zeros((len(data),num_labels))
    for i,cat in enumerate(list(data.label.values)):
        y_vector[i][cat] = 1

    # todo
    y_image_adds = None
    return data, y_vector, y_image_adds

    


# loss : https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/clip/modeling_clip.py#L69
# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def compute_batch_dot_product(x_t,x_v):
    print("xt", x_t.size())
    print("xv", x_v.size())
    x_t = normalize(x_t, dim=1)
    x_v = normalize(x_v, dim=1)
    m_batchsize, _ = x_t.size()
    d = torch.bmm(x_t.view(m_batchsize,1,fixed_feat_size),x_v.view(m_batchsize,fixed_feat_size,1))
    print("d", d.size())
    d = torch.squeeze(d)
    return d

def loss_correction(T,loss_fn, output, label):
    # https://arxiv.org/pdf/2102.05336.pdf
    print("loss correction")
    #print("label", label.size()) 
    T = torch.tensor(T)
    #print("T", T.size())
    eneg, one_minus_eneg = T[0][1], T[0][0]
    eplus, one_minus_eplus = T[1][0], T[1][1]
    # compute loss for each class
    label_zero = torch.tensor([[1.,0.]]*label.size()[0])
    #print("label_zero", label_zero.size())
    loss_zero = loss_fn(output,label_zero)
    #print("loss_zero", loss_zero.size())
    label__one = torch.tensor([[0.,1.]]*label.size()[0])
    #print("label_one", label__one.size())
    loss_one = loss_fn(output,label__one)
    #print("loss_one", loss_one.size())
    # compute loss
    targets = torch.argmax(label, dim=1)
    loss = torch.zeros(targets.size()[0])
    for i in range(targets.size()[0]):
        if targets[i] == 0:
            loss[i] = one_minus_eplus * loss_zero[i] - eneg * loss_one[i]
        else:
            # label == 1
            loss[i] = one_minus_eneg * loss_one[i] - eplus *loss_zero[i]
        loss[i] = loss[i]/(one_minus_eplus-eneg)

    #print("loss", loss)
    #print("loss", loss.size())
    loss = loss.mean()
    #print("loss", loss)
    #print("loss", loss.size())
    return loss

# class TIRLoss(nn.Module):
#     def __init__(self, weight):
#         super(TIRLoss, self).__init__()
#         self.weight = weight

#     def forward(self, output, target, similarity_score):
#         print("out",output.size())
#         print("target",target.size())    
#         loss = F.cross_entropy(output,target,weight=self.weight,reduction="none")
#         print("loss",loss.size())
#         print("loss",loss.mean())
#         loss = loss - similarity_score
#         loss = loss.mean()
#         print("new loss", loss)
#         return loss

# class CustomLoss(nn.Module):
#     def __init__(self, pos_weight):
#         super(CustomLoss, self).__init__()
#         self.pos_weight = pos_weight
#         self.criterion =nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)


#     def forward(self, output, target, x_t, x_v):
#         #target = torch.LongTensor(target)
        
#         loss = self.criterion(output, target)
#         print("xt", x_t.size())
#         print("xv", x_v.size())
#         x_t = normalize(x_t, dim=1)
#         x_v = normalize(x_v, dim=1)
#         m_batchsize, _ = x_t.size()
#         d = torch.bmm(x_t.view(m_batchsize,1,fixed_feat_size),x_v.view(m_batchsize,fixed_feat_size,1))
#         print("d", d.size())
#         d = torch.squeeze(d).mean()
#         print("d",d)
    
#         print("loss",loss)
#         loss = loss + d
#         print("new loss", loss)
        
        
#         return loss

def get_optimizer_params(named_parameters, weight_decay, lr, verbose = False):
        optimizer_params = []
        params = {'lr':lr, 'weight_decay':weight_decay}
        params['params'] = []
        if verbose:
            print("requires grad params:")
        for name, param in named_parameters:
            if verbose:
                print(name)
            if param.requires_grad == True:
                params['params'].append(param)
        optimizer_params.append(params)
        return optimizer_params

def compute_metrics(res, num_classes, multi_label=False):
    if not multi_label:
   
        metrics = {
            "f1_weighted": F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
            "f1_macro" : F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "precision_weighted": Precision(task="multiclass", num_classes=num_classes, average="weighted"),
            "precision_macro" : Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall_weighted": Recall(task="multiclass", num_classes=num_classes, average="weighted"),
            "recall_macro" : Recall(task="multiclass", num_classes=num_classes, average="macro"),
            }
    else:
        metrics = {
            "f1_weighted": F1Score(task="multilabel", num_classes=num_classes, average="weighted"),
            "f1_macro" : F1Score(task="multilabel", num_classes=num_classes, average="macro"),
            "precision_weighted": Precision(task="multilabel", num_classes=num_classes, average="weighted"),
            "precision_macro" : Precision(task="multilabel", num_classes=num_classes, average="macro"),
            "recall_weighted": Recall(task="multilabel", num_classes=num_classes, average="weighted"),
            "recall_macro" : Recall(task="multilabel", num_classes=num_classes, average="macro"),
            }
    
    y, y_pred = res["labels"], res["predictions"]
    results = {}
    for name,metric in metrics.items():
        results[name] = metric(y_pred, y).item()
    results["loss"] = res["loss"]
    metric_list, res_list = [],[]
    for metric, res in results.items():
        metric_list.append(metric)
        res_list.append(res)
    results = {"metric":metric_list,"result":res_list}    
    return results

def agg_metrics_val(res_val, metric_names, num_labels):
    metrics_val = {}
    metrics_val["metric"] = metric_names
    for predictions in res_val:
        metrics = compute_metrics(predictions,num_labels)
        # make sure metric list is in the same order for all epochs
        metric_dict = dict(zip(metrics["metric"],metrics["result"]))
        metrics_val["epoch-"+str(predictions["epoch"]+1)] = [metric_dict[metric] for metric in metric_names]
    return metrics_val
   

