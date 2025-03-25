import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_privacy(model, test_data, tul_classifier):
    """Evaluate privacy protection using TUL-based metrics"""
    model.eval()
    tul_classifier.eval()
    
    # Generate synthetic trajectories
    with torch.no_grad():
        # Convert test data to tensors and move to device
        real_tensors = {k: torch.stack([torch.from_numpy(t[k]).float() for t in test_data]).to(model.device) 
                       for k in test_data[0].keys()}
        
        # Generate synthetic data
        noise = torch.randn(len(test_data), model.latent_dim, device=model.device)
        syn_tensors = model.generator(noise)
        
        # Get predictions from TUL classifier
        real_pred = tul_classifier(real_tensors)
        syn_pred = tul_classifier(syn_tensors)
        
        # Convert to numpy for metric computation
        real_pred = real_pred.cpu().numpy()
        syn_pred = syn_pred.cpu().numpy()
        
        # Get true labels (assuming they are indices in the test data)
        real_labels = np.arange(len(test_data))
    
    # Compute metrics
    metrics = {}
    
    # ACC@1 (Top-1 accuracy)
    real_acc1 = np.mean(np.argmax(real_pred, axis=1) == real_labels)
    syn_acc1 = np.mean(np.argmax(syn_pred, axis=1) == real_labels)
    metrics['acc1'] = {
        'real': real_acc1,
        'syn': syn_acc1,
        'improvement': real_acc1 - syn_acc1
    }
    
    # ACC@5 (Top-5 accuracy)
    real_acc5 = np.mean([real_labels[i] in np.argsort(real_pred[i])[-5:] 
                        for i in range(len(real_pred))])
    syn_acc5 = np.mean([real_labels[i] in np.argsort(syn_pred[i])[-5:] 
                       for i in range(len(syn_pred))])
    metrics['acc5'] = {
        'real': real_acc5,
        'syn': syn_acc5,
        'improvement': real_acc5 - syn_acc5
    }
    
    # Macro Precision
    real_macro_p = precision_score(real_labels, 
                                 np.argmax(real_pred, axis=1), 
                                 average='macro')
    syn_macro_p = precision_score(real_labels, 
                                np.argmax(syn_pred, axis=1), 
                                average='macro')
    metrics['macro_p'] = {
        'real': real_macro_p,
        'syn': syn_macro_p,
        'improvement': real_macro_p - syn_macro_p
    }
    
    # Macro Recall
    real_macro_r = recall_score(real_labels, 
                               np.argmax(real_pred, axis=1), 
                               average='macro')
    syn_macro_r = recall_score(real_labels, 
                              np.argmax(syn_pred, axis=1), 
                              average='macro')
    metrics['macro_r'] = {
        'real': real_macro_r,
        'syn': syn_macro_r,
        'improvement': real_macro_r - syn_macro_r
    }
    
    # Macro F1
    real_macro_f1 = f1_score(real_labels, 
                            np.argmax(real_pred, axis=1), 
                            average='macro')
    syn_macro_f1 = f1_score(real_labels, 
                           np.argmax(syn_pred, axis=1), 
                           average='macro')
    metrics['macro_f1'] = {
        'real': real_macro_f1,
        'syn': syn_macro_f1,
        'improvement': real_macro_f1 - syn_macro_f1
    }
    
    return metrics 