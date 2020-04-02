package edu.uw.bhi.bionlp.covid.parser;

import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierOuterClass.Sentence;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifierclient.AssertionClassifierClient;
import edu.uw.bhi.bionlp.covid.parser.data.UMLSConcept;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetamapLiteParser;

import java.util.List;
import java.util.ArrayList;

public class NoteProcessor {

    MetamapLiteParser parser = new MetamapLiteParser();
    AssertionClassifierClient assertionClassifier = new AssertionClassifierClient();

    public PipelineResult processNote(String note) {

        PipelineResult result = new PipelineResult();
        
        
        /* 
         * Chunk into sentences.
         */
        List<Sentence> sentences = new ArrayList<Sentence>();
        try {
            sentences = assertionClassifier.detectSentences(note);
        } catch (Exception ex) {
            System.out.println("Error: failed to chunk sentences. " + ex.getMessage());
        }

        /*
         * For each sentence.
         */
        for (int sID = 0; sID < sentences.size(); sID++) {
            /* 
             * Extract UMLS concepts.
             */
            List<UMLSConcept> concepts = new ArrayList<UMLSConcept>();
            try {
                concepts = parser.parseSentenceWithMetamap(sentences.get(sID).getText());
            } catch (Exception ex) {
                System.out.println("Error: failed to parse with Metamap. Sentence: '" + sentences.get(sID).getText() + "'. " + ex.getMessage());
            }

            /* 
             * For each concept, if indexes valid, predict assertion.
             */
		    for (UMLSConcept concept : concepts) {
                String prediction = "";
                try {
                    if (concept.getBeginTokenIndex() == 0 && concept.getEndTokenIndex() == 0) {
                        prediction = "present";
                    } else {
                        prediction = assertionClassifier.predictAssertion(sentences.get(sID).getText(), concept.getBeginTokenIndex(), concept.getEndTokenIndex());
                    }
                } catch (Exception ex) {
                    System.out.println("Error: failed to predict concept. Concept: '" + concept.getConceptName() + "'. " + ex.getMessage());
                } finally {
                    // sentence.concepts.add(new PipelineConcept(concept, prediction));
                }
            }
            // result.sentences.add(sentence);
        }
        return result;
    }

    class PipelineResult {
        List<PipelineSentence> sentences = new ArrayList<PipelineSentence>();
    
        public List<PipelineSentence> getSentences() {
            return sentences;
        }
    }
    
    class PipelineSentence {
        int index;
        String text;
        List<PipelineConcept> concepts = new ArrayList<PipelineConcept>();
    
        public PipelineSentence(int index, String text) {
            this.index = index;
            this.text = text;
        }
    
        public int getIndex() { 
            return index;
        }
    
        public String getText() { 
            return text;
        }
    
        public List<PipelineConcept> getConcepts() {
            return concepts;
        }
    }
    
    class PipelineConcept extends UMLSConcept {
        String prediction;
    
        public PipelineConcept(UMLSConcept concept, String prediction) {
            this.setCUI(concept.getCUI());
            this.setConceptName(concept.getConceptName());
            this.setPhrase(concept.getPhrase());
            this.setSemanticTypeLabels(concept.getSemanticTypeLabels());
            this.setBeginTokenIndex(concept.getBeginTokenIndex());
            this.setEndTokenIndex(concept.getEndTokenIndex());
            this.setPrediction(prediction);
        }
    
        public String getPrediction() {
            return prediction;
        }
    
        public void setPrediction(String prediction) {
            this.prediction = prediction;
        }
    }
}