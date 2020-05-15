# covid-parser

**covid-parser** is an ongoing project by the Bio-NLP group at the University of Washington led by Professor Meliha Yetisgen. 
The goal is to create a sophisticated natural language processing tool suite for identifying COVID diagnostic and testing status, 
as well as signs and symptoms. 

As our work is ongoing, the structure and code within this repo will change rapidly.

## MetaMapLite Installation
We use MetaMapLite for baseline evaluation. Note that because [MetaMapLite](https://metamap.nlm.nih.gov/MetaMapLite.shtml) is a 
product of the NIH/NLM and requires an active (though free) UMLS license, we can't distribute it as a part of our source code. Therefore in order to use the MetaMapLite Docker container here you must follow the below steps:

1. Obtain a UMLS license, if you don't already have one. (You'll be prompted to enter your login or create one when downloading the files in step 2.
2. Under the `Downloads` section of the MetaMapLite site, download `MetaMapLite 2018 3.6.2rc3 binaryonly Version` and `2018AB UMLS Level 0+4+9 Dataset` zip files. Unzip each of them.
3. Open `public_mm_lite_3.6.2rc3_binaryonly` and copy the jar file `/public_mm_lite/target/metamaplite-3.6.2rc3-standalone.jar` to `/covid-parser/metamap/lib/nlm/metamaplite/3.6.2rc3-standalone/`.
4. Also within `public_mm_lite_3.6.2rc3_binaryonly`, copy `/public_mm_lite/data/models` and `/public_mm_lite/data/specialterms.txt` to `/covid-parser/metamap/resources/public_mm_lite/data/`.

