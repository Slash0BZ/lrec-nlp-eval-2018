package org.cogcomp;

import edu.illinois.cs.cogcomp.annotation.BasicTextAnnotationBuilder;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Constituent;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.TextAnnotation;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.View;
import edu.illinois.cs.cogcomp.pos.POSAnnotator;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.System.exit;
import static java.lang.System.in;

/**
 * Created by xuany on 2/5/2018.
 */
public class POSEvaluator {

    public static final String TestFile = "data/wsj-00";

    public static List<Pair<List<String>, List<String>>> readData(String filename){
        List<Pair<List<String>, List<String>>> ret = new ArrayList<>();
        try {
            File file = new File(filename);
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = bufferedReader.readLine()) != null){
                List<String> POSTags = new ArrayList<>();
                List<String> Tokens = new ArrayList<>();
                Matcher m = Pattern.compile("\\((.*?)\\)").matcher(line);
                while (m.find()){
                    String pair = m.group(1);
                    String[] pairGroup = pair.split(" ");
                    POSTags.add(pairGroup[0]);
                    Tokens.add(pairGroup[1]);
                }
                ret.add(new Pair<>(POSTags, Tokens));
                if (POSTags.size() != Tokens.size()){
                    exit(-1);
                }
            }
        }
        catch (Exception e){

        }
        return ret;
    }

    public static void cogcompNLP(){
        List<Pair<List<String>, List<String>>> sentences = readData(TestFile);
        POSAnnotator posAnnotator = new POSAnnotator();
        Map<String, Integer> goldMap = new HashMap<>();
        Map<String, Integer> predictMap = new HashMap<>();
        Map<String, Integer> correctMap = new HashMap<>();
        for (Pair<List<String>, List<String>> sentence : sentences){
            List<String[]> tokenizedSentence = new ArrayList<>();
            String[] tokensArr = new String[sentence.getSecond().size()];
            tokensArr = sentence.getSecond().toArray(tokensArr);
            tokenizedSentence.add(tokensArr);
            TextAnnotation ta =
                    BasicTextAnnotationBuilder.createTextAnnotationFromTokens(" ", " ", tokenizedSentence);
            try {
                ta.addView(posAnnotator);
            }
            catch (Exception e){
                e.printStackTrace();
            }
            View TokenView = ta.getView(ViewNames.POS);
            for (int i = 0; i < TokenView.getNumberOfConstituents(); i++){
                String goldTag = sentence.getFirst().get(i);
                String predictedTag = TokenView.getConstituentsCoveringToken(i).get(0).getLabel();
                incrementMap(goldMap, goldTag);
                incrementMap(predictMap, predictedTag);
                if (goldTag.equals(predictedTag)){
                    incrementMap(correctMap, goldTag);
                }
            }
        }
        Map<String, Pair<Double, Double>> result = producePerformence(goldMap, predictMap, correctMap);
        for (String key : result.keySet()){
            System.out.println(key + ": " + result.get(key).getFirst() + ", " + result.get(key).getSecond());
        }
    }

    public static void CoreNLP(){
        List<Pair<List<String>, List<String>>> sentences = readData(TestFile);
        Map<String, Integer> goldMap = new HashMap<>();
        Map<String, Integer> predictMap = new HashMap<>();
        Map<String, Integer> correctMap = new HashMap<>();
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        for (Pair<List<String>, List<String>> sent : sentences){
            String[] tokensArr = new String[sent.getSecond().size()];
            tokensArr = sent.getSecond().toArray(tokensArr);
            List<CoreMap> tokenizedSentence = new ArrayList<>();
            CoreMap sentence = new ArrayCoreMap();
            List<CoreLabel> words = SentenceUtils.toCoreLabelList(tokensArr);
            sentence.set(CoreAnnotations.TokensAnnotation.class, words);
            tokenizedSentence.add(sentence);
            Annotation document = new Annotation(tokenizedSentence);
            pipeline.annotate(document);

            List<CoreMap> annotatedSentences = document.get(CoreAnnotations.SentencesAnnotation.class);
            int index = 0;
            int sentenceIndex = 0;
            for (CoreMap annotatedSentence : annotatedSentences){
                sentenceIndex ++;
                if (sentenceIndex > 1){
                    continue;
                }
                int tokenCount = annotatedSentence.get(CoreAnnotations.TokensAnnotation.class).size();
                if (tokenCount != sent.getSecond().size()){
                    continue;
                }
                for (CoreLabel token : annotatedSentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    String predictedTag = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                    String goldTag = sent.getFirst().get(index);
                    incrementMap(goldMap, goldTag);
                    incrementMap(predictMap, predictedTag);
                    if (goldTag.equals(predictedTag)){
                        incrementMap(correctMap, goldTag);
                    }
                    index ++;
                }
            }
        }
        Map<String, Pair<Double, Double>> result = producePerformence(goldMap, predictMap, correctMap);
        for (String key : result.keySet()){
            System.out.println(key + ": " + result.get(key).getFirst() + ", " + result.get(key).getSecond());
        }
    }

    public static void incrementMap(Map<String, Integer> target, String key){
        if (target.containsKey(key)){
            target.put(key, target.get(key) + 1);
        }
        else {
            target.put(key, 1);
        }
    }

    public static Map<String, Pair<Double, Double>> producePerformence(Map<String, Integer> goldMap, Map<String, Integer> predictMap, Map<String, Integer> correctMap){
        Set<String> labelSet = new HashSet<>(goldMap.keySet());
        Set<String> labelSetPredict = new HashSet<>(predictMap.keySet());
        labelSet.addAll(labelSetPredict);

        Map<String, Pair<Double, Double>> ret = new HashMap<>();
        for (String label : labelSet){
            int predict = predictMap.containsKey(label) ? predictMap.get(label) : 0;
            int labeled = goldMap.containsKey(label) ? goldMap.get(label) : 0;
            int correct = correctMap.containsKey(label) ? correctMap.get(label) : 0;
            double precision = (double)correct / (double) predict;
            double recall = (double)correct / (double)labeled;
            ret.put(label, new Pair<>(precision, recall));
        }
        return ret;
    }

    public static void main(String[] args){
        CoreNLP();
    }
}
