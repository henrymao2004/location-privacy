import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_privacy(model, test_data, tul_classifier):
    """Evaluate privacy protection using TUL-based metrics"""
    model.eval()
    tul_classifier.eval()
    
    # Generate synthetic trajectories
    with torch.no_grad():
        noise = torch.randn(len(test_data), model.latent_dim, device=model.device)
        gen_trajs = model.generator([*test_data[:4], test_data[4], noise])
        
        # Get predictions from TUL classifier
        real_pred = tul_classifier(test_data)
        gen_pred = tul_classifier(gen_trajs)
        
        # Convert to numpy for metric computation
        real_pred = real_pred.cpu().numpy()
        gen_pred = gen_pred.cpu().numpy()
    
    # Compute metrics
    metrics = {}
    
    # ACC@1 (Top-1 accuracy)
    real_acc1 = np.mean(np.argmax(real_pred, axis=1) == np.argmax(real_pred, axis=1))
    gen_acc1 = np.mean(np.argmax(gen_pred, axis=1) == np.argmax(real_pred, axis=1))
    metrics['acc1'] = {
        'real': real_acc1,
        'gen': gen_acc1,
        'improvement': real_acc1 - gen_acc1
    }
    
    # ACC@5 (Top-5 accuracy)
    real_acc5 = np.mean([np.argmax(real_pred[i]) in np.argsort(real_pred[i])[-5:] 
                        for i in range(len(real_pred))])
    gen_acc5 = np.mean([np.argmax(gen_pred[i]) in np.argsort(real_pred[i])[-5:] 
                       for i in range(len(gen_pred))])
    metrics['acc5'] = {
        'real': real_acc5,
        'gen': gen_acc5,
        'improvement': real_acc5 - gen_acc5
    }
    
    # Macro Precision
    real_macro_p = precision_score(np.argmax(real_pred, axis=1), 
                                 np.argmax(real_pred, axis=1), 
                                 average='macro')
    gen_macro_p = precision_score(np.argmax(gen_pred, axis=1), 
                                np.argmax(real_pred, axis=1), 
                                average='macro')
    metrics['macro_p'] = {
        'real': real_macro_p,
        'gen': gen_macro_p,
        'improvement': real_macro_p - gen_macro_p
    }
    
    # Macro Recall
    real_macro_r = recall_score(np.argmax(real_pred, axis=1), 
                               np.argmax(real_pred, axis=1), 
                               average='macro')
    gen_macro_r = recall_score(np.argmax(gen_pred, axis=1), 
                              np.argmax(real_pred, axis=1), 
                              average='macro')
    metrics['macro_r'] = {
        'real': real_macro_r,
        'gen': gen_macro_r,
        'improvement': real_macro_r - gen_macro_r
    }
    
    # Macro F1
    real_macro_f1 = f1_score(np.argmax(real_pred, axis=1), 
                            np.argmax(real_pred, axis=1), 
                            average='macro')
    gen_macro_f1 = f1_score(np.argmax(gen_pred, axis=1), 
                           np.argmax(real_pred, axis=1), 
                           average='macro')
    metrics['macro_f1'] = {
        'real': real_macro_f1,
        'gen': gen_macro_f1,
        'improvement': real_macro_f1 - gen_macro_f1
    }
    
    return metrics 