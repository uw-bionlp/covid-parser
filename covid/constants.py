from collections import OrderedDict





'''
Labeling
'''
BEGIN = 'B-'
INSIDE = 'I-'
OUTSIDE = 'O'
UNKNOWN = 'unknown'
NEG_LABEL = 0

# Result naming File naming conventions
SINGLE_STAGE = 'single-stage'
END_TO_END = 'end-to-end'
EVAL_RANGE = 'eval_range'

MICRO = 'micro'
MACRO = 'macro'

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
QC = 'qc'
RANDOM = 'random'
ACTIVE = 'active'
MANUAL = 'manual'

STATUS_VALUE = 'StatusValue'
ENTITY_VALUE = 'EntityValue'
STATUS_ARG = 'StatusArg'
ENTITY_ARG = 'EntityArg'
BINARY = 'Binary'
STATUS_SEQ = 'Status_seq'
STATUS_SPAN = 'Status_span'

INDICATOR = 'Indicator'
INDICATOR_SEQ = 'Indicator_seq'


VALUE = 'Value'

'''
Determants
'''

DETERMINANT = 'Determinant'

ALCOHOL = 'Alcohol'
COUNTRY = 'Country'
DRUG = 'Drug'
EMPLOYMENT = 'Employment'
ENVIRO_EXPOSURE = 'EnviroExposure'
GENDER_ID = 'GenderID'
INSURANCE = 'Insurance'
LIVING_STATUS = 'LivingStatus'
PHYS_ACTIVITY = 'PhysActivity'
RACE = 'Race'
SEXUAL_ORIENT = 'SexualOrient'
TOBACCO = 'Tobacco'
SSX = 'SSx'

OCCUPATION = 'Occupation'
ENVIRONMENTAL_EXPOSURE = 'EnvironmentalExposure'
LIVING_SIT = 'LivingSituation'
PHYSICAL_ACTIVITY = 'PhysicalActivity'


SUBSTANCES = [ALCOHOL, DRUG, TOBACCO]

'''
Entities
'''

TRIGGER = 'Trigger'


# Span and class - new
STATUS_TIME             = 'StatusTime'
STATUS_TIME_VAL         = 'StatusTimeVal'

DEGREE                  = 'Degree'
DEGREE_VAL              = 'DegreeVal'

STATUS_EMPLOY           = 'StatusEmploy'
STATUS_EMPLOY_VAL       = 'StatusEmployVal'

STATUS_INSURE           = 'StatusInsure'
STATUS_INSURE_VAL       = 'StatusInsureVal'

TYPE_GENDER_ID          = 'TypeGenderID'
TYPE_GENDER_ID_VAL      = 'TypeGenderIDVal'

TYPE_LIVING             = 'TypeLiving'
TYPE_LIVING_VAL         = 'TypeLivingVal'

TYPE_SEXUAL_ORIENT      = 'TypeSexualOrient'
TYPE_SEXUAL_ORIENT_VAL  = 'TypeSexualOrientVal'

# Span and class - previous
STATUS = 'Status'
STATE = 'State'

# Span only - new
AMOUNT      = 'Amount'
DURATION    = 'Duration'
FREQUENCY   = 'Frequency'
HISTORY     = 'History'
METHOD      = 'Method'
TYPE        = 'Type'

# Span only - previous
EXPOSURE_HISTORY = 'ExposureHistory'
QUIT_HISTORY = 'QuitHistory'
LOCATION = 'Location'


QUANTITY = 'Quantity'
SUBSTANCE = 'Substance'

SEQ_TAGS = 'Seq_tags'

ENTITIES = 'Entities'
ENTITY = 'Entity'
EVENTS = 'events'
SEQ_LABELS = 'seq_labels'
SENT_LABELS = 'sent_labels'

LAB_TYPE = 'label_type'
LAB_DEF = 'label_definition'
LAB_MAP = 'label_map'
SEQ = 'sequence'
SENT = 'sentence'

EVAL_TYPE = 'Evaluation type'
EVENT_TYPE = 'Event type'
SPAN_TYPE = 'Span type'
EVENT_SPAN_TYPE = 'Event span type'
EVENT_LABEL = 'event_label'
LABEL = 'Label'
ROUND = 'Round'
COUNT = 'Count'
ARGUMENT = 'Argument'
SPAN = 'Span'
TOKENS = 'Tokens'


SPAN_ONLY = 'Span-only argument'
LABELED = 'Labeled argument'

CURRENT = 'current'
PAST = 'past'
NONE = 'none'
FUTURE = 'future'
STATUS_HIERARCHY = [OUTSIDE, NONE, FUTURE, PAST, CURRENT]

EMPLOYED = 'employed'
UNEMPLOYED = 'unemployed'
RETIRED = 'retired'
ON_DISABILITY = 'on_disability'
STUDENT = 'student'
HOMEMAKER = 'homemaker'

ALONE = 'alone'
WITH_FAMILY = 'with_family'
WITH_OTHERS = 'with_others'
HOMELESS = 'homeless'

MULTILABEL = 'multilabel'

SOCIAL = 'social'
HPI = 'hpi'









'''
Sentence processing
'''
START_TOKEN = '<s>'
END_TOKEN = '</s>'
UNK = '<unk>'

'''
File naming conventions
'''

# Generated files
PREDICTIONS_FILE = 'predictions.pkl'
GOLD_FILE = 'gold_labels.pkl'
SWEEPS_FILE = 'sweeps.csv'
SCORES_FILE_XLSX = 'scores.xlsx'
SCORES_FILE = 'scores.csv'
SCORES_ALL = 'scores_all.csv'
SCORES_BY_EVENT = 'scores_by_event.csv'
SCORES_SUMMARY = 'scores_summary.csv'
MODEL_FILE = 'model.pkl'
CORPUS_FILE = 'corpus.pkl'
HYPERPARAMS_FILE = 'hyperparams.json'
FEATPARAMS_FILE = 'featparams.json'
DESCRIP_FILE = 'descrip.json'
STATE_DICT = 'state_dict.pt'
CONFIG = 'config.json'

# BRAT files
ANNOTATION_CONF = 'annotation.conf'
VISUAL_CONF = 'visual.conf'
TEXT_FILE_EXT = 'txt'
ANN_FILE_EXT = 'ann'

# BERT
BERT_VOCAB = 'vocab.txt'
BERT_CONFIG = 'bert_config.json'
BERT_CHECKPOINT = 'bert_model.ckpt'


BERT = 'bert'
XLNET = 'xlnet'

ENCODING = 'utf-8'

NONE_PH = '--'


NT = 'NT'
NP = 'NP'
TP = 'TP'
TN = 'TN'
FP = 'FP'
FN = 'FN'
P = 'P'
R = 'R'
F1 = 'F1'


SINGLE = 'single'
CV_TUNE = 'cv_tune'
CV_PREDICT = 'cv_predict'
TRAIN = 'train'
PREDICT = 'predict'

MIMIC = 'mimic'
UW = 'uw'
UW_OPIOID_ADMIT = 'uw_opioid_admit'
UW_OPIOID_ED = 'uw_opioid_ed'
UW_FLU = 'uw_flu'
SOCIAL_DET = 'social_det'
YVNOTES = 'yvnotes'
COVID = 'COVID'
LUNG_CANCER = 'lung_cancer'



ID = 'id'
SUBSET = 'subset'
SOURCE = 'source'
SOURCE_FINE = 'source_fine'
SELECTION = 'selection'
GENERAL = 'general'
INITIAL = 'initial'
CATEGORY = 'category'
NOTE_TYPE = 'note_type'

AGREEMENT = 'agreement'

TAG_MAP = OrderedDict()

# Sources
TAG_MAP[MIMIC] = SOURCE 
TAG_MAP[UW] = SOURCE 
TAG_MAP[SOCIAL_DET] = SOURCE 
TAG_MAP[YVNOTES] = SOURCE 
TAG_MAP[LUNG_CANCER] = SOURCE
TAG_MAP[COVID] = SOURCE

TAG_MAP[UW_OPIOID_ADMIT] = SOURCE_FINE 
TAG_MAP[UW_OPIOID_ED] = SOURCE_FINE 
TAG_MAP[UW_FLU] = SOURCE_FINE 

# Note types
TAG_MAP[AGREEMENT] = AGREEMENT

# Data subsets
TAG_MAP[TRAIN] = SUBSET 
TAG_MAP[DEV] = SUBSET 
TAG_MAP[TEST] = SUBSET 
TAG_MAP[QC] = SUBSET 
    
# Selection types
TAG_MAP[MANUAL] = SELECTION
TAG_MAP[RANDOM] = SELECTION 
TAG_MAP[ACTIVE] = SELECTION 
    
# Categories
TAG_MAP[INITIAL] = CATEGORY 
TAG_MAP[GENERAL] = CATEGORY        

TAG_MAP['round01'] = ROUND
TAG_MAP['round02'] = ROUND
TAG_MAP['round03'] = ROUND
TAG_MAP['round03a'] = ROUND
TAG_MAP['round03b'] = ROUND
TAG_MAP['round04'] = ROUND
TAG_MAP['round05'] = ROUND
TAG_MAP['round06'] = ROUND
TAG_MAP['round07'] = ROUND
TAG_MAP['round08'] = ROUND
TAG_MAP['round09'] = ROUND
TAG_MAP['round10'] = ROUND
TAG_MAP['round11'] = ROUND
TAG_MAP['round12'] = ROUND
TAG_MAP['round13'] = ROUND
TAG_MAP['round14'] = ROUND
TAG_MAP['round15'] = ROUND
TAG_MAP['round16'] = ROUND
TAG_MAP['round17'] = ROUND
TAG_MAP['round18'] = ROUND
TAG_MAP['round19'] = ROUND
TAG_MAP['round20'] = ROUND

RND = 'rnd'
TAG_MAP['RND01'] = RND
TAG_MAP['RND02'] = RND
TAG_MAP['RND03'] = RND
TAG_MAP['RND04'] = RND
TAG_MAP['RND05'] = RND
TAG_MAP['RND06'] = RND
TAG_MAP['RND07'] = RND
TAG_MAP['RND08'] = RND
TAG_MAP['RND09'] = RND
TAG_MAP['RND10'] = RND
TAG_MAP['RND11'] = RND
TAG_MAP['RND12'] = RND

POSITIVE = 'positive'
NEGATIVE = 'negative'
TEST_RESULT = 'test_result'
TAG_MAP[POSITIVE] = TEST_RESULT
TAG_MAP[NEGATIVE] = TEST_RESULT


TESTED = 'tested'
NOT_TESTED = 'not_tested'
TEST_STATUS = 'test_status'
TAG_MAP[TESTED] = TEST_STATUS
TAG_MAP[NOT_TESTED] = TEST_STATUS


SHORT = 'short'
LONG = 'long'
LENGTH = 'length'
TAG_MAP[SHORT] = LENGTH
TAG_MAP[LONG] = LENGTH

REQUIRED = 'required'

SPAN_GOLD = 'gold'
SPAN_CRF = 'crf'
SPAN_PRUNE = 'prune'
SPAN_FFNN = 'ffnn'
ARGUMENT_GOLD = 'gold'
ARGUMENT_RULE = 'rule'
ARGUMENT_LEARNED = 'learned'

# Models
MULTITASK = 'multitask'
SPAN_EXTRACT = 'span_extract'

OVERLAPS = 'overlaps'

# Distance metrics
SINE = 'sine'


DESCRIP = 'Description'