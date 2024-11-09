# import torch 
# from torch import nn
# import timm
# from transformers import DistilBertModel, DistilBertConfig , ViTModel
# # # import config as config
# from torch.nn import functional as F
# # from module import ImageEncoder,TextEncoder,ProjectionHead


# #-------------------------Image Encoder-------------------------#
# class ImageEncoder(nn.Module):
#     """
#     Encodes images to a fixed vector size representation.
#     Args:
#         Input: Image url or Image
#         Output: Fixed Latent Representation.
#     Note:
#         Official Paper says we can use Resnet50 or Vision-Transformer.
#     """
#     def __init__(self, model_name=config.vision_encoder_model, pretrained=config.pretrained, trainable=config.trainable):
#         super().__init__()
        
#         self.model = timm.create_model(model_name=model_name,pretrained=pretrained,num_classes=0,global_pool="avg")
#         # self.model = ViTModel() #Incase of using a Vision Transformer

#         for p in self.model.parameters():
#             p.requires_grad = trainable
    
#     def forward(self,x):
#         return self.model(x)
    

# #-------------------------Text Encoder-------------------------#
# class TextEncoder(nn.Module):
#     """
#     Encodes Text into fixed vector size representation.
#     Args:
#         Input: Text
#         Output : Fixed Latent Representation
#     Note:
#         Official Paper says we can use CBOW or Text Transformers.
#     """
#     def __init__(self,model_name=config.text_encoder_model,pretrained=config.pretrained,trainable=config.trainable):
#         super().__init__()

#         if pretrained:
#             self.model = DistilBertModel.from_pretrained(model_name)
#         else:
#             self.model = DistilBertModel(config=DistilBertConfig())

#         for p in self.model.parameters():
#             p.requires_grad = trainable

#         # we are using the CLS token hidden representation as the sentence's embedding
#         self.target_token_idx = 0

#     def forward(self,input_ids,attention_mask):
#         output = self.model(input_ids=input_ids,attention_mask=attention_mask)
#         last_hidden_state = output.last_hidden_state
#         return last_hidden_state[:, self.target_token_idx, :]


# #-------------------------Projection Head-------------------------#
# class ProjectionHead(nn.Module):
#     """
#     Projection Head performs linear projection followed by non-linear activation and another linear layer with dropout and layer normalization
#     Args : 
#         Input : Embedding_dim, Projection_dim , Dropout
#         Outputs : Projection output of given input
#     """
#     def __init__(self,embedding_dim,projection_dim=config.projection_dim,dropout=config.dropout):
#         super().__init__()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(projection_dim)

#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x += projected
#         return self.layer_norm(x)




# #-------------------CLIP------------------#
# class CLIPModel(nn.Module):
#     """
#     Implementation of CLIP Architecture.
#     Args:
#         Input  :
#         Output :
#     Note :
#     """

#     # Define these
#     def __init__(self,temperature,image_embedding,text_embedding):
#         super().__init__()
#         self.image_encoder = ImageEncoder()
#         self.text_encoder = TextEncoder()
#         self.text_proj = ProjectionHead(embedding_dim=text_embedding)
#         self.image_proj = ProjectionHead(embedding_dim=image_embedding)
#         self.temperature = temperature
    
#     def forward(self,batch):
#         # Get Image and Text features
#         image_features = self.image_encoder(batch['image'])
#         text_features = self.text_encoder(input_ids=batch["input_ids"],attention_mask=batch['attention_mask'])

#         # Getting the Image and Text features to same dim.
#         image_embeddings = self.image_proj(image_features)
#         text_embeddings = self.text_proj(text_features)

#         # Contrastive Loss (@ used for matrix multiplication)
#         logits = (text_embeddings @ image_embeddings.T) / self.temperature
#         image_similarity = image_embeddings @ image_embeddings.T
#         text_similarity = text_embeddings @ text_embeddings.T
#         targets = F.softmax(
#             (image_similarity + text_similarity)/(2*self.temperature) , dim=1 
#             )
#         texts_loss = F.cross_entropy(logits, targets, reduction='none')
#         images_loss = F.cross_entropy(logits.T,targets.T,reduction='none')
#         loss = ((images_loss+texts_loss)/2.0)
#         return loss.mean()
    
# # #----------------------Main-----------------#
# # if __name__ == '__main__':
# #     images = torch.randn(8, 3, 224, 224)
# #     input_ids = torch.randint(5, 300, size=(8, 25))
# #     attention_mask = torch.ones(8, 25)
# #     batch = {
# #         'image': images,
# #         'input_ids': input_ids,
# #         'attention_mask': attention_mask
# #     }

# #     CLIP = CLIPModel()
# #     loss = CLIP(batch)
# #     print("CLIP")