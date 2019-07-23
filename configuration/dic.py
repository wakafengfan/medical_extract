tag_dictionary = {
    'O': 0,

    'B_disease': 1,
    'I_disease': 2,

    'B_diagnosis': 3,
    'I_diagnosis': 4,

    'B_symptom': 5,
    'I_symptom': 6,

    'B_drug': 7,
    'I_drug': 8,

    # '<START>': 9,
    # '<STOP>': 10
}

trans_list = ['O', 'B_disease', 'I_disease', 'B_diagnosis', 'I_diagnosis', 'B_symptom', 'I_symptom', 'B_drug', 'I_drug']

sub_dic = {
    'O': 'O',

    'disease-B': 'B_disease',
    'disease-I': 'I_disease',

    'diagnosis-B': 'B_diagnosis',
    'diagnosis-I': 'I_diagnosis',

    'symptom-B': 'B_symptom',
    'symptom-I': 'I_symptom',

    'drug-B': 'B_drug',
    'drug-I': 'I_drug'

}

trans = {
    '00': 0.2,

    '01': 0.2,
    '12': 0.1,
    '22': 0.1,
    '20': 0.2,
    '21': 0.1,
    '23': 0.1,
    '25': 0.1,
    '27': 0.1,

    '03': 0.2,
    '34': 0.1,
    '44': 0.1,
    '40': 0.2,
    '41': 0.1,
    '43': 0.1,
    '45': 0.1,
    '47': 0.1,

    '05': 0.2,
    '56': 0.1,
    '66': 0.1,
    '60': 0.2,
    '65': 0.1,
    '61': 0.1,
    '63': 0.1,
    '67': 0.1,

    '07': 0.2,
    '78': 0.1,
    '88': 0.1,
    '80': 0.2,
    '81': 0.1,
    '83': 0.1,
    '85': 0.1,
    '87': 0.1,
}



tag_dictionary_start = {
    'O': 0,

    'B_disease': 1,

    'B_diagnosis': 2,

    'B_symptom': 3,

    'B_drug': 4,
}


tag_dictionary_stop = {
    'O': 0,

    'I_disease': 1,

    'I_diagnosis': 2,

    'I_symptom': 3,

    'I_drug': 4,
}