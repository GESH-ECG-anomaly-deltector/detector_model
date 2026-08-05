# necessary libraries and packages:

import sys 
import os
import wfdb
import scipy.io
import numpy as np
import neurokit2 as nk
import csv
import shutil









class load_dataset():
    def __init__(self):

        self.log_flag=1
        print("Do you want to see the logs?")
        self.log_flag=int(input("1-Yes         0-No"))

        self.log_path="".join(["logs/", "load_dataset"])
        ls_list=os.listdir()
        if "logs" in ls_list:
            shutil.rmtree("logs")
        os.mkdir("logs")
        os.mkdir(self.log_path)
        print("\n\n\n\n")



    #def comment_function(block_num, main_massage, hint="", hint_list=[]):
    def comment_function(self, function_name, main_massage, hint="", hint_list=[]):
        func_log_name=[self.log_path, "/", function_name, ".txt"]
        func_log_name="".join(func_log_name)

        log_text=[]
        log_text.append(function_name)
        log_text.append("\n  ")
        log_text.append(main_massage)
        log_text.append("\n")
        if(len(hint_list)>0):
            log_text.append("\thint:\n")
            log_text.append("\t\t ")
            log_text.append(hint)
            log_text.append(" :\n")
            for hint_item in hint_list:
                log_text.append("\t\t\t -> ")
                log_text.append(hint_item)
                log_text.append("\n")
        log_text.append("----------------------------------------------------------------")

        log_text="".join(log_text)

        with open(func_log_name , "w") as file:
            file.write(log_text)
        
        if(self.log_flag!=0):
            print(log_text, "\n\n\n\n")



    def load_dataset(self):
        signal_array=[]
        record_array=[]

        counter=10      #we limit our dataset to ease the process:
        counter=int(input(" -->Enter how many instances you want to load: "))
        i=0

        #previous path for niusha project:
        #directory1="../../../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"

        directory1="../../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"
        ls1=os.listdir(directory1)

        for item1 in ls1:
            if(i>=counter):
                break
        
            directory2=[directory1, item1]
            directory2="/".join(directory2)
    
            ls2=os.listdir(directory2)

            for item2 in ls2:
                if(i>=counter):
                    break
                
                directory3=[directory2, item2]
                directory3="/".join(directory3)
        
                ls3=os.listdir(directory3)

                directory4="/".join([directory3, "RECORDS"])

                file_names=[]
                with open(directory4, "r") as f:
                    for line in f:
                        file_names.append(line.strip())
        
                for name in file_names:
                    print("num:", i, end="\r")
            
                    if(i>=counter):
                        break
            
                    directory5="/".join([directory3, name])

                    mat_file=".".join([directory5, "mat"])
                    hea_file=".".join([directory5, "hea"])

                    mat_data=scipy.io.loadmat(mat_file)
                    signal=mat_data["val"]
                    #this signal has 12 leads

                    record=wfdb.rdheader(directory5)

                    signal_array.append(signal)
                    record_array.append(record)

                    i+=1

        self.signals=np.array(signal_array)
        self.record_array=np.array(record_array)


        self.comment_function(
            function_name="load_dataset()",
            main_massage="".join(["first ", str(counter), " instances of the dataset are loaded"])
        )



    def denoise_signals(self):
        clean_signal_array=[]

        i=0
        for item in self.signals:
            print("num:", i, end="\r")
            tmp_array=[]
            for item2 in item:
                item3=nk.ecg_clean(item2)
                tmp_array.append(item3)
            tmp_array=np.array(tmp_array)
            clean_signal_array.append(tmp_array)
            i+=1
        self.signals=np.array(clean_signal_array)

        self.signals = self.signals.transpose(0, 2, 1)

        self.comment_function(
            function_name="denoise_signal()", 
            main_massage="All instances are denoised by Neurokit2.", 
            hint="signals_shape=(batch_number, time_steps, features) as bellow:",
            hint_list=[str(self.signals.shape)]
        )



    def reduce_sample_rate(self):    
        self.signals=self.signals[:, ::5, :]

        self.comment_function(
            function_name="reduce_sample_rate()",
            main_massage="sample-rate is reduced from 500HZ to 100HZ.", 
            hint="signals_shape=(batch_number, timesteps, features) as bellow",
            hint_list=[str(self.signals.shape)]
        )



    def define_labels(self):
        #define labels from record_array:

        labels=[]
        for item in self.record_array:
            labels.append(item.comments[2])

        tmp_labels=[]
        for item in labels:
            tmp=item.split("Dx: ")
            tmp_labels.append(tmp[1])

        labels=np.array(tmp_labels)

        tmp_labels=[]
        for item1 in labels:
            tmp=item1.split(",")
            tmp_labels.append(tmp)

        self.labels=tmp_labels

        self.comment_function(
            function_name="define_labels()",
            main_massage="The labels of the instances are extracted from self.record_array."
        )


    
    def define_8_superclasses(self):
        #definition of the 8 superclasses:
        
        directory1="../../a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/ConditionNames_SNOMED-CT.csv"
        lines=[]
        with open(directory1, "r") as f:
            for line in f:
                lines.append(line.strip())

        tmp_lines=[]
        for line in lines:
            tmp_lines.append(line.split(","))

    
        #data format in tmp_line is as bellow:
        #Acronym Name	Full Name	Snomed_CT
        tmp_line=tmp_lines[0]
        del tmp_lines[0]

        lines=np.array(tmp_lines)


        super_classes=[]

        #0
        tmp_labels=[]
        tmp_labels.append(lines[53])
        tmp_class=[0, "Standard", tmp_labels]
        super_classes.append(tmp_class)

        #1
        tmp_labels=[]
        tmp_labels.extend(lines[0:5])
        tmp_labels.append(lines[10])
        tmp_labels.append(lines[17])
        tmp_labels.append(lines[31])
        tmp_labels.append(lines[52])
        tmp_class=[1, "Bradyarrhythmia", tmp_labels]
        super_classes.append(tmp_class)

        #2
        tmp_labels=[]
        tmp_labels.extend(lines[26:31])
        tmp_labels.extend(lines[37:44])
        tmp_class=[2, "Ischemia and ST-T changes", tmp_labels]
        super_classes.append(tmp_class)

        #3
        tmp_labels=[]
        tmp_labels.extend(lines[44:48])
        tmp_labels.append(lines[49])
        tmp_class=[3, "Ventricular arrhythmia", tmp_labels]
        super_classes.append(tmp_class)

        #4
        tmp_labels=[]
        tmp_labels.append(lines[5])
        tmp_labels.append(lines[7])
        tmp_labels.append(lines[18])
        tmp_labels.extend(lines[54:63])
        tmp_class=[4, "Supraventricular arrhythmia", tmp_labels]
        super_classes.append(tmp_class)

        #5
        tmp_labels=[]
        tmp_labels.extend(lines[15:17])
        tmp_labels.extend(lines[19:23])
        tmp_labels.extend(lines[35:37])
        tmp_class=[5, "Bundle branch abnormalities", tmp_labels]
        super_classes.append(tmp_class)

        #6
        tmp_labels=[]
        tmp_labels.append(lines[6])
        tmp_labels.extend(lines[8:10])
        tmp_labels.extend(lines[23:26])
        tmp_labels.extend(lines[33:35])
        tmp_class=[6, "Abnormal parameters in ECG", tmp_labels]
        super_classes.append(tmp_class)

        #7
        tmp_labels=[]
        tmp_labels.append(lines[48])
        tmp_labels.append(lines[51])
        tmp_class=[7, "Pre-excitation", tmp_labels]
        super_classes.append(tmp_class)

        #8
        tmp_labels=[]
        tmp_labels.extend(lines[11:15])
        tmp_labels.append(lines[32])
        tmp_labels.append(lines[50])
        tmp_class=[8, "delete_all_of_them", tmp_labels]
        super_classes.append(tmp_class)

        
        self.superclasses=super_classes
           
        log_text=[]
        log_text.append("superclasses are as bellow\n\n\n")
        log_text.append("super-classes with their sub-classes\n\n\n")
        for super_class in self.superclasses:
            log_text.append("\n-----------------\n")
            log_text.append(str(super_class[0]))
            log_text.append("  ")
            log_text.append(super_class[1])
            log_text.append("\n")
            for line in super_class[2]:
                log_text.append("\t")
                log_text.append(line[0])
                log_text.append(",\t")
                log_text.append(line[1])
                log_text.append("\n\n")
            log_text.append("\n\n")


        log_text="".join(log_text)


        self.comment_function(
            function_name="define_8_superclasses()", 
            main_massage="8 superclasses based on Dr.Teimouri's suggestion are created.", 
            hint="super_classes_log",
            hint_list=[log_text]
        )



    def set_index(self, label_array, label):
        for i_0 in range(len(self.superclasses)): 
            tmp_number0=self.superclasses[i_0][0]
    
            i_1=2
            for i_2 in range(len(self.superclasses[i_0][i_1])):
                i_3=2
                tmp_number1=self.superclasses[i_0][i_1][i_2][i_3]
                if(label==tmp_number1):
                    label_array[tmp_number0]=1
        return label_array



    def one_hot_encodding(self): 
        tmp_final_labels=[]
 
        for label in self.labels:
            tmp_labels=[]
            for i in range(9):
                tmp_labels.append(0)

            for item in label:
                tmp_labels=self.set_index(tmp_labels, item)

            # deletting super_class_7:
            flag=1
            if tmp_labels[8]==1 and tmp_labels[0]!=1:
                flag=0
                for i in range(len(tmp_labels)):
                    if i!=8  and  tmp_labels[i]==1:
                        flag=1
                        break;
            if(flag==0):
                tmp_labels[0]=1
            del tmp_labels[8]

            tmp_final_labels.append(tmp_labels)

        self.labels=np.array(tmp_final_labels)

        dld_log=self.delete_labelless_data()

        self.comment_function(
            function_name="one_hot_encodding()", 
            main_massage="one-hot encodding functions are applied on the dataset",
            hint="outputs",
            hint_list=
            [
                "signals: numpy array that includes 12-lead ECG signal.",
                "labels : one-hot encodded numpy array of diagnosed labels.", 
                dld_log
            ]
        )



    def delete_labelless_data(self):
        remove_indices=[]
        for i in range(len(self.labels)):
            if not 1 in self.labels[i]:
                remove_indices.append(i)

        self.signals=np.delete(self.signals, remove_indices, axis=0)
        self.labels =np.delete(self.labels,  remove_indices, axis=0)

        log_text=[]
        log_text.append("The signals with this indices have no label, so they are romoved:\n\t\t\t\t")
        log_text.append(str(remove_indices))
        log_text.append("\n\n")
        log_text="".join(log_text)

        return log_text



    def get_signals_and_labels(self):
        return self.signals, self.labels