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
    '00': 0.5,

    '01': 0.5,
    '12': 0.5,
    '22': 0.5,
    '20': 0.5,
    '21': 0.5,
    '23': 0.5,
    '25': 0.5,
    '27': 0.5,

    '03': 0.5,
    '34': 0.5,
    '44': 0.5,
    '40': 0.5,
    '41': 0.5,
    '43': 0.5,
    '45': 0.5,
    '47': 0.5,

    '05': 0.5,
    '56': 0.5,
    '66': 0.5,
    '60': 0.5,
    '65': 0.5,
    '61': 0.5,
    '63': 0.5,
    '67': 0.5,

    '07': 0.5,
    '78': 0.5,
    '88': 0.5,
    '80': 0.5,
    '81': 0.5,
    '83': 0.5,
    '85': 0.5,
    '87': 0.5,
}



tag_dictionary_start = {
    'O': 0,

    'B_disease': 1,

    'B_diagnosis': 2,

    'B_symptom': 3,

    'B_drug': 4,
}


tag_dictionary_stopt = {
    'O': 0,

    'I_disease': 1,

    'I_diagnosis': 2,

    'I_symptom': 3,

    'I_drug': 4,
}