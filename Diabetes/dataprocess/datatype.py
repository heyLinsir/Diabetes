import numpy as np

class BasicData(object):
    def __init__(self, name):
        super(BasicData, self).__init__()
        self.name = name

    def map(self, value):
        '''
        map :
            value from diabetic_data.csv for a specific key/name
            -->
            statistical continuous value(float or list of float)
        '''
        raise NotImplementedError

    def __call__(self, data):
        '''
        data  : an example of patient
        return: statistical continuous numpy value
        '''
        value = self.map(data[self.name])
        if type(value) == float:
            value = [value]
        return np.asarray(value, 'float32')

class encounter_id(BasicData):
    def __init__(self):
        super(encounter_id, self).__init__('encounter_id')
        
    def map(self, value):
        return float(value)

class patient_nbr(BasicData):
    def __init__(self):
        super(patient_nbr, self).__init__('patient_nbr')
        
    def map(self, value):
        return float(value)

class race(BasicData):
    def __init__(self):
        super(race, self).__init__('race')

        self.value_dict = {'?': 0, 'other': 0}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            self.value_dict[value] = max(self.value_dict.values()) + 1
        one_hot = [0.] * (max(self.value_dict.values()) + 1)
        one_hot[self.value_dict[value]] = 1.
        return one_hot

class gender(BasicData):
    def __init__(self):
        super(gender, self).__init__('gender')

        self.value_dict = {'?': 0, 'female': 1, 'male': 2}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            return [1., 0., 0.]
        one_hot = [0., 0., 0.]
        one_hot[self.value_dict[value]] = 1.
        return one_hot

class age(BasicData):
    def __init__(self):
        super(age, self).__init__('age')
        
    def map(self, value):
        begin, end = value.split('[')[1].split(')')[0].split('-')
        return 0.5 * (float(begin) + float(end))

class weight(BasicData):
    def __init__(self):
        super(weight, self).__init__('weight')
        
    def map(self, value):
        '''
        return [x, y]
        x=1 for unknown weight, x=0 for known weight
        y=0 for unknown weight
        '''
        if value == '?':
            return [1., 0.]
        else:
            return [0., float(value)]

class admission_type_id(BasicData):
    def __init__(self):
        super(admission_type_id, self).__init__('admission_type_id')
        
    def map(self, value):
        value = int(value)
        one_hot = [0.] * (value + 1)
        one_hot[value] = 1.
        return one_hot

class discharge_disposition_id(BasicData):
    def __init__(self):
        super(discharge_disposition_id, self).__init__('discharge_disposition_id')
        
    def map(self, value):
        value = int(value)
        one_hot = [0.] * (value + 1)
        one_hot[value] = 1.
        return one_hot

class admission_source_id(BasicData):
    def __init__(self):
        super(admission_source_id, self).__init__('admission_source_id')
        
    def map(self, value):
        value = int(value)
        one_hot = [0.] * (value + 1)
        one_hot[value] = 1.
        return one_hot

class time_in_hospital(BasicData):
    def __init__(self):
        super(time_in_hospital, self).__init__('time_in_hospital')
        
    def map(self, value):
        return float(value)

class payer_code(BasicData):
    def __init__(self):
        super(payer_code, self).__init__('payer_code')
        
    def map(self, value):
        '''
        return [x, y]
        x=1 for unknown payer_code, x=0 for known payer_code
        y=0 for unknown payer_code
        '''
        if value == '?':
            return [1., 0.]
        else:
            return [0., float(value)]

class num_lab_procedures(BasicData):
    def __init__(self):
        super(num_lab_procedures, self).__init__('num_lab_procedures')
        
    def map(self, value):
        return float(value)

class num_procedures(BasicData):
    def __init__(self):
        super(num_procedures, self).__init__('num_procedures')
        
    def map(self, value):
        return float(value)  
        
class num_medications(BasicData):
    def __init__(self):
        super(num_medications, self).__init__('num_medications')
        
    def map(self, value):
        return float(value)

class number_outpatient(BasicData):
    def __init__(self):
        super(number_outpatient, self).__init__('number_outpatient')
        
    def map(self, value):
        return float(value) 

class number_emergency(BasicData):
    def __init__(self):
        super(number_emergency, self).__init__('number_emergency')
        
    def map(self, value):
        return float(value) 

class number_inpatient(BasicData):
    def __init__(self):
        super(number_inpatient, self).__init__('number_inpatient')
        
    def map(self, value):
        return float(value)         

class diag_1(BasicData):
    '''
    unfinished
    '''
    def __init__(self):
        super(diag_1, self).__init__('diag_1')
        
    def map(self, value):
        return [0] 

class diag_2(BasicData):
    '''
    unfinished
    '''
    def __init__(self):
        super(diag_2, self).__init__('diag_2')
        
    def map(self, value):
        return [0] 

class diag_3(BasicData):
    '''
    unfinished
    '''
    def __init__(self):
        super(diag_3, self).__init__('diag_3')
        
    def map(self, value):
        return [0]  

class number_diagnoses(BasicData):
    def __init__(self):
        super(number_diagnoses, self).__init__('number_diagnoses')
        
    def map(self, value):
        return float(value)  

class max_glu_serum(BasicData):
    def __init__(self):
        super(max_glu_serum, self).__init__('max_glu_serum')

        # maybe distributed value is better than one hot representation?
        self.value_dict = {'none': 0, 'normal': 1, '>200': 2, '>300': 3}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'none'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot     

class A1Cresult(BasicData):
    def __init__(self):
        super(A1Cresult, self).__init__('A1Cresult')

        # maybe distributed value is better than one hot representation?
        self.value_dict = {'none': 0, 'normal': 1, '>7': 2, '>8': 3}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'none'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot  

class features_for_medications(BasicData):
    def __init__(self):
        super(features_for_medications, self).__init__('features_for_medications')

        self.value_dict = {'no': 0, 'up': 1, 'steady': 2, 'down': 3}
        self.key_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                        'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
                        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                        'metformin-rosiglitazone', 'metf+Y1:AU1ormin-pioglitazone']
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'no'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot 

    def __call__(self, data):
        '''
        data  : an example of patient
        return: statistical continuous numpy value
        '''
        one_hot = []
        for key in self.key_list:
            one_hot.extend(self.map(data[key]))
        return np.asarray(value, 'float32')        

class change(BasicData):
    def __init__(self):
        super(change, self).__init__('change')

        self.value_dict = {'no': 0, 'change': 1}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'no'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot  

class diabetesMed(BasicData):
    def __init__(self):
        super(diabetesMed, self).__init__('diabetesMed')

        self.value_dict = {'no': 0, 'yes': 1}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'no'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot   

class readmitted(BasicData):
    def __init__(self):
        super(readmitted, self).__init__('readmitted')

        self.value_dict = {'no': 0, '<30': 1, '>30': 2}
        
    def map(self, value):
        value = value.lower()
        if value not in self.value_dict:
            value = 'no'
        one_hot = [0.] * len(self.value_dict)
        one_hot[self.value_dict[value]] = 1.
        return one_hot                                                     
