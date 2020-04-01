package edu.uw.bhi.bionlp.covid.parser.metamap;

import bioc.BioCDocument;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import gov.nih.nlm.nls.metamap.document.FreeText;
import gov.nih.nlm.nls.metamap.lite.types.ConceptInfo;
import gov.nih.nlm.nls.metamap.lite.types.Entity;
import gov.nih.nlm.nls.metamap.lite.types.Ev;
import gov.nih.nlm.nls.ner.MetaMapLite;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.lang3.StringUtils;

/**
 *
 * @author wlau
 * @author ndobb
 */
public class MetamapLiteParser implements IMetamapParser {

    Properties myProperties;
    MetaMapLite metaMapLiteInst;

    static String[] semanticTypes = {"acab", "anab", "comd", "cgab", "dsyn", "emod", "fndg", "inpo", "mobd", "neop", "patf", "sosy"};
    static Set<String> semanticTypesToBeIncluded = new HashSet<String>(Arrays.asList(semanticTypes));

    public MetamapLiteParser() {
        try {
            myProperties = new Properties();
            MetaMapLite.expandModelsDir(myProperties,
                    "resources/public_mm_lite/data/models");
            MetaMapLite.expandIndexDir(myProperties,
                    "resources/public_mm_lite/data/ivf/2018ABascii/USAbase");
            myProperties.setProperty("metamaplite.excluded.termsfile",
                    "resources/public_mm_lite/data/specialterms.txt");

            metaMapLiteInst = new MetaMapLite(myProperties);

        } catch (Exception ex) {
            Logger.getLogger(MetamapLiteParser.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println(ex.getMessage());
            System.out.println(ex.getStackTrace());
        }
    }

    public List<Entity> parseSentenceWithMetamap(BioCDocument document) throws Exception {
        return metaMapLiteInst.processDocument(document);
    }

    public List<UMLSConcept> parseSentenceWithMetamap(String sentenceText) throws Exception {
        BioCDocument document = FreeText.instantiateBioCDocument(sentenceText);
        document.setID("1");

        List<UMLSConcept> concepts = new ArrayList<UMLSConcept>();
        String pretokenized = sentenceText.trim().replaceAll("\\s+", " ");
        String[] sentenceTokens = pretokenized.split("\\s+");
        
        int beginTokenIndex = -1;
        int endTokenIndex = 0;
        int nextTokenIndex = 0;
        List<Entity> resultList = metaMapLiteInst.processDocument(document);

        for (Entity entity : resultList) {
            String phraseText = entity.getMatchedText();
            String[] phraseTokens = phraseText.split("\\s+");
            beginTokenIndex = -1;

            // Find the begin token index of the phrase
            for (int k = nextTokenIndex; k < sentenceTokens.length; k++) {
                boolean match = true;
                for (int j = 0; match && j<phraseTokens.length && j+k<sentenceTokens.length; j++) {
                    String sentenceToken = StringUtils.strip(sentenceTokens[j+k]," [!\"#$%&'()*+,-./:;<=>?\\\\@\\[\\]^_`{|}~0-9]");
                    match = sentenceToken.equalsIgnoreCase(phraseTokens[j]);
                }
                if (match) {
                    beginTokenIndex = k; 
                    endTokenIndex = beginTokenIndex + phraseTokens.length - 1;
                    nextTokenIndex = beginTokenIndex;
                    break;
                }
            }
            List<String> cuiList = new ArrayList<String>();

            for (Ev ev : entity.getEvSet()) {

                // filter based on semantic types
                boolean flag = true;
                ConceptInfo conceptInfo = ev.getConceptInfo();
                List<String> SemanticTypeList = new ArrayList<String>(conceptInfo.getSemanticTypeSet());
                for (String semType : SemanticTypeList) {
                    flag = semanticTypesToBeIncluded.contains(semType); 
                    if (flag) {
                        break;
                    }
                }

                if (flag == false) {
                    continue;
                }

                String semanticTypes = "";
                if (conceptInfo.getSemanticTypeSet().size() == 1) {
                    semanticTypes = SemanticTypeList.get(0);
                } else {
                    for (String semType : SemanticTypeList) {
                        semanticTypes += semType + "|";
                    }
                }

                if (!cuiList.contains(conceptInfo.getCUI()) && flag == true) {
                    UMLSConcept concept = new UMLSConcept();
                    concept.setCUI(conceptInfo.getCUI());
                    concept.setConceptName(conceptInfo.getPreferredName());
                    concept.setPhrase(phraseText);
                    concept.setBeginTokenIndex(beginTokenIndex);
                    concept.setEndTokenIndex(endTokenIndex);
                    concept.setSemanticTypeLabels(semanticTypes);
                    concepts.add(concept);
                }
                cuiList.add(conceptInfo.getCUI());
            }
        }
        return concepts;
    }
}