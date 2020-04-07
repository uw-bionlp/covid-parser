package edu.uw.bhi.bionlp.covid.parser.metamap;

import bioc.BioCDocument;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import gov.nih.nlm.nls.metamap.document.FreeText;
import gov.nih.nlm.nls.metamap.lite.types.ConceptInfo;
import gov.nih.nlm.nls.metamap.lite.types.Entity;
import gov.nih.nlm.nls.metamap.lite.types.Ev;
import gov.nih.nlm.nls.ner.MetaMapLite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author wlau
 * @author ndobb
 */
public class MetamapLiteParser implements IMetamapParser {

    Properties myProperties;
    MetaMapLite metaMapLiteInst;

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
        List<Entity> resultList = metaMapLiteInst.processDocument(document);

        for (Entity entity : resultList) {
            String matchedText = entity.getMatchedText();
            int charStartIdx = entity.getStart();
            int charStopIdx = charStartIdx + entity.getLength();
            HashMap<String,Boolean> cuiList = new HashMap<String,Boolean>();

            for (Ev ev : entity.getEvSet()) {
                ConceptInfo conceptInfo = ev.getConceptInfo();
                String cui = conceptInfo.getCUI();
                String semanticTypes = "";
                List<String> SemanticTypeList = new ArrayList<String>(conceptInfo.getSemanticTypeSet());
                if (conceptInfo.getSemanticTypeSet().size() == 1) {
                    semanticTypes = SemanticTypeList.get(0);
                } else {
                    for (String semType : SemanticTypeList) {
                        semanticTypes += semType + "|";
                    }
                }

                if (cuiList.get(cui) == null) {
                    UMLSConcept concept = new UMLSConcept();
                    concept.setCUI(cui);
                    concept.setConceptName(conceptInfo.getPreferredName());
                    concept.setPhrase(matchedText);
                    concept.setBeginCharIndex(charStartIdx);
                    concept.setEndCharIndex(charStopIdx);
                    concept.setSemanticTypeLabels(semanticTypes);
                    concepts.add(concept);
                }
                cuiList.put(cui, true);
            }
        }
        return concepts;
    }
}