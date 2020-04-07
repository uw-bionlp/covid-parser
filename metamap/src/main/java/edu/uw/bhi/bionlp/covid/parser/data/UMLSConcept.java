package edu.uw.bhi.bionlp.covid.parser.data;

/**
 * Author: melihay
 * Date: Jun 28, 2012
 * Time: 11:31:09 AM
 * Version: 1.0
 */
public class UMLSConcept {
    private String CUI;
    private String conceptName;
    private String phrase; // phrase the concept identified in
    private String semanticTypeLabels;
    private int beginTokenIndex;
    private int endTokenIndex;
    private int beginCharIndex;
    private int endCharIndex;

    public String toString() {
        String str = "";
        str += this.getCUI()+"\t"+"PHRASE:"+this.getPhrase()+"\tSTR:"+this.getConceptName()+"\tSEMTYPES:"+this.getSemanticTypeLabels();
        return str;
    }

    public String getCUI() {
        return CUI;
    }

    public void setCUI(String CUI) {
        this.CUI = CUI;
    }

    public String getPhrase() {
        return phrase;
    }

    public void setPhrase(String phrase) {
        this.phrase = phrase;
    }

    public String getConceptName() {
        return conceptName;
    }

    public void setConceptName(String conceptName) {
        this.conceptName = conceptName;
    }

    public int getBeginTokenIndex() {
        return beginTokenIndex;
    }

    public void setBeginTokenIndex(int beginTokenIndex) {
        this.beginTokenIndex = beginTokenIndex;
    }

    public int getEndTokenIndex() {
        return endTokenIndex;
    }

    public void setEndTokenIndex(int endTokenIndex) {
        this.endTokenIndex = endTokenIndex;
    }

    public int getBeginCharIndex() {
        return beginCharIndex;
    }

    public void setBeginCharIndex(int beginCharIndex) {
        this.beginCharIndex = beginCharIndex;
    }

    public int getEndCharIndex() {
        return endCharIndex;
    }

    public void setEndCharIndex(int endCharIndex) {
        this.endCharIndex = endCharIndex;
    }

    public String getSemanticTypeLabels() {
        return semanticTypeLabels;
    }

    public void setSemanticTypeLabels(String semanticTypeLabels) {
        this.semanticTypeLabels = semanticTypeLabels;
    }

    public String getNgram(String sentence, int ngramSize) {
        int lastChar = sentence.length()-1;
        int precCnt = 0;
        int follCnt = 0;
        int startPos = this.beginCharIndex;
        int endPos = this.endCharIndex;
        boolean isSpace = false;
        boolean prevWasSpace = true;
        String sent = sentence.trim().replaceAll("\\s+", " ");

        // Get preceding
        while (startPos > -1) {
            isSpace = sentence.charAt(startPos) == ' ';
            if (isSpace && !prevWasSpace) {
                precCnt++;
            }
            if (startPos == 0 || precCnt > ngramSize) {
                if (precCnt > ngramSize) {
                    precCnt = ngramSize;
                }
                break;
            }
            startPos--;
            prevWasSpace = isSpace;
        }

        this.setBeginTokenIndex(precCnt);
        this.setEndTokenIndex(this.beginTokenIndex + this.phrase.split(" ").length - 1);

        // Get following
        prevWasSpace = true;
        while (endPos <= lastChar) {
            isSpace = sentence.charAt(endPos) == ' ';
            if (isSpace && !prevWasSpace) {
                follCnt++;
            }
            if (endPos == lastChar || follCnt == ngramSize) {
                break;
            }
            endPos++;
            prevWasSpace = isSpace;
        }
        return sentence.substring(startPos, endPos).trim();
    }
}