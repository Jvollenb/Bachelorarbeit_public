from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.losses import FocalLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import yaml
import pandas as pd
from adversarials.attack_fgsm import MultiInputFGSM
from adversarials.attack_pgd import PGD_MultiInput


warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)


    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        self.last_epoch = 0
        print("Total class: ",self.args.num_class)
        # model init
        print(f"DEBUG (Exp_Classification): Initializing model '{self.args.model}' with:")
        print(f"  args.d_model: {self.args.d_model}")
        print(f"  args.enc_in: {self.args.enc_in}")
        print(f"  feature_df.shape: {train_data.feature_df.shape}")
        
        
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='CE':
            criterion = nn.CrossEntropyLoss()
        elif self.args.loss=='FOCAL':
            cls_weights=np.load(self.args.root_path+'/'+'TRAIN_cls_weight.npy')
            print("Class weights: ")
            print(cls_weights)
            cls_weights = torch.FloatTensor(cls_weights).to(self.device)
            criterion=FocalLoss(alpha=cls_weights)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_cwt, batch_x_another,label) in enumerate(vali_loader):
                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device)

                outputs = self.model(batch_x_cwt,batch_x_another)  
                loss = criterion(outputs, label.long().squeeze())

                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        attack_generator = PGD_MultiInput(
            self.model, 
            eps=self.args.train_eps, 
            steps=10
        )

        # attack_generator = MultiInputFGSM(
        #     self.model, 
        #     eps=self.args.train_eps
        # )

        self.last_epoch = self.args.train_epochs

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x_cwt,batch_x_another, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device).squeeze(-1)
                # print(batch_x.shape)
                # print(label.shape)
                self.model.eval()

                adv_x_cwt, adv_x_another = attack_generator(batch_x_cwt, batch_x_another, label)

                if self.args.white_noise > 0:
                    noise = torch.randn_like(adv_x_another) * self.args.white_noise
                    adv_x_another = adv_x_another + noise
                    noise = torch.randn_like(adv_x_cwt) * self.args.white_noise
                    adv_x_cwt = adv_x_cwt + noise

                # 4. Schalte das Modell zurück in den train()-Modus für den Trainingsschritt
                self.model.train()
                model_optim.zero_grad()
                outputs_adv = self.model(adv_x_cwt,adv_x_another)
                loss_adv = criterion(outputs_adv, label.long().squeeze(-1))

                
                loss_adv.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                if self.args.white_noise > 0:
                    noise = torch.randn_like(batch_x_another) * self.args.white_noise
                    batch_x_another = batch_x_another + noise
                    noise = torch.randn_like(batch_x_cwt) * self.args.white_noise
                    batch_x_cwt = batch_x_cwt + noise

                #model_optim.zero_grad()
                
                outputs_clean = self.model(batch_x_cwt,batch_x_another)
                # print(outputs.shape)
                loss_clean = criterion(outputs_clean, label.long().squeeze(-1))
                loss_clean.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()
                total_loss = (1 - self.args.adversarial_weight) * loss_clean + self.args.adversarial_weight * loss_adv
                
                # total_loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # model_optim.step()

                train_loss.append(loss_clean.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self.last_epoch = epoch
                break
            if (epoch + 1) % 50 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print("----------------------------------------------------")
        print(f"The Model took {time.time() - time_now}sec to train")
        print("----------------------------------------------------")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # preds = []
        # trues = []
       
        self.model.eval()
        
        # Map string names from the config file to the actual attack classes
        attack_class_map = {
            "MultiInputFGSM": MultiInputFGSM,
            "PGD_MultiInput": PGD_MultiInput,
        }

        # Load evaluation jobs from the YAML config file
        with open(self.args.attack_config, 'r') as f:
            config_jobs = yaml.safe_load(f)

        # Build the final evaluation_jobs list by replacing string names with class objects
        evaluation_jobs = []
        for job in config_jobs:
            if job['attack'] is not None:
                job['attack'] = attack_class_map[job['attack']]
            evaluation_jobs.append(job)

        results = {}

        for job in evaluation_jobs:
            job_name = job["name"]
            attack = job["attack"]

            print(f"--- Running Evaluation: {job_name} ---")
    
            total_correct = 0
            total_samples = 0

            attack_instance = None
            if job["attack"] is not None:
                # Erstelle die Angriffs-Instanz: sehr sauber!
                attack_instance = job["attack"](self.model, **job["params"])

            for i, (batch_x_cwt,batch_x_another, label) in enumerate(test_loader):

                batch_x_cwt = batch_x_cwt.float().to(self.device)
                batch_x_another =batch_x_another.float().to(self.device)

                label = label.to(self.device).squeeze(-1)

                # Initialisiere die finalen Inputs mit den sauberen Daten
                final_x_cwt = batch_x_cwt
                final_x_another = batch_x_another

                # 2. Führe den Angriff aus, falls definiert
                if job["attack"] is not None:
                    # Konvertiere zu Numpy NUR für den ART-Aufruf
                    # x_cwt_np = batch_x_cwt.cpu().numpy()
                    # x_another_np = batch_x_another.cpu().numpy()
                    # input_tuple = (x_cwt_np, x_another_np)
                    
                    # Instanziiere den Angriff
                    # attack_instance = job["attack"](**job["params"])
                    
                    # Generiere die adversariellen Daten
                    # adv_input_tuple = attack_instance.generate(x=input_tuple)
                    
                    # # Konvertiere das Ergebnis DIREKT wieder zu Tensoren
                    # # Dies ist der einzige Punkt, an dem die Konvertierung stattfindet.
                    # final_x_cwt = torch.from_numpy(adv_input_tuple[0]).to(self.device)
                    # final_x_another = torch.from_numpy(adv_input_tuple[1]).to(self.device)

                    final_x_cwt, final_x_another = attack_instance(batch_x_cwt, batch_x_another, label)

                    
                with torch.no_grad():
                    outputs = self.model(final_x_cwt, final_x_another)

                # 4. Berechne die Genauigkeit
                _, predicted = torch.max(outputs.data, 1)
                total_samples += label.size(0)
                total_correct += (predicted == label.squeeze()).sum().item()
    
            accuracy = total_correct / total_samples
            results[job_name] = accuracy
            print(f"--- Accuracy for {job_name}: {accuracy * 100:.2f}% --- \n")
        
        print("========== FINAL EVALUATION RESULTS ==========")
        for name, acc in results.items():
            print(f"{name:<30s} | Accuracy: {acc * 100:.2f}%")
        print("============================================")
        
        # Erstelle ein Dictionary für die einzelne Ergebniszeile, das alle Hyperparameter enthält
        row_data = {
            'Seed': self.args.random_seed,
            'Model': self.args.model,
            'dropout': self.args.dropout,
            'train_epochs': self.args.train_epochs,
            'train_eps': self.args.train_eps,
            'last_epoch': self.last_epoch,
            'white_noise': self.args.white_noise,
            'batch': self.args.batch_size,
            'patience': self.args.patience,
            'LR': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
            'd_model': self.args.d_model,
            'n_heads': self.args.n_heads,
            'e_fact': self.args.e_fact,
            'dconv': self.args.dconv,
            'dstate': self.args.d_state,
            'projected_space': self.args.projected_space,
            'checkpoint_path': setting,
            'mambas': self.args.num_mambas,
            'initial focus': self.args.initial_focus,
            'channel_token_mixing': self.args.channel_token_mixing,
            'no_rocket': self.args.no_rocket,
            'max_pooling': self.args.max_pooling,
            'half_rocket': self.args.half_rocket,
            'additive_fusion': self.args.additive_fusion,
            'only_forward_scan': self.args.only_forward_scan,
            'reverse_flip': self.args.reverse_flip,
            'variation': self.args.variation,
            'comment': self.args.comment
        }

        # Füge die Genauigkeit jedes Angriffs als neue Spalte hinzu
        for attack_name, acc in results.items():
            # Bereinige den Angriffsnamen, um einen gültigen und sauberen Spaltennamen zu erhalten
            clean_name = "Acc_" + attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_").replace("=", "").replace(".", "")
            row_data[clean_name] = acc

        # Erstelle einen DataFrame aus dem Dictionary mit der einen Zeile
        temp_df = pd.DataFrame([row_data])

        csv_path = './csv_results/classification/' + 'result_' + self.args.model_id + '.csv'
        if not os.path.exists(csv_path):
            temp_df.to_csv(csv_path, index=False)
        else:
            result_df = pd.read_csv(csv_path)
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
            result_df.to_csv(csv_path, index=False)
        return
