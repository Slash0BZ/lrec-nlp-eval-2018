package org.cogcomp;

import edu.illinois.cs.cogcomp.annotation.BasicTextAnnotationBuilder;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
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
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.System.exit;

/**
 * Created by xuany on 2/5/2018.
 */
public class POSEvaluator {

    public static final String TestFile = "data/00-23.br";

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
        Map<String, Pair<Double, Double>> result = producePerformance(goldMap, predictMap, correctMap);
        System.out.println("==========CogcompNLP POS PERFORMANCE==========");
        printPerformance(result);
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
        Map<String, Pair<Double, Double>> result = producePerformance(goldMap, predictMap, correctMap);
        System.out.println("==========CoreNLP POS PERFORMANCE==========");
        printPerformance(result);
    }

    public static void OpenNLP_maxent(){
        List<Pair<List<String>, List<String>>> sentences = readData(TestFile);
        Map<String, Integer> goldMap = new HashMap<>();
        Map<String, Integer> predictMap = new HashMap<>();
        Map<String, Integer> correctMap = new HashMap<>();
        POSModel posModel = null;
        POSTaggerME tagger = null;
        try {
            InputStream posModelIn = new FileInputStream("OpenNLPModels/en-pos-maxent.bin");
            posModel = new POSModel(posModelIn);
            tagger = new POSTaggerME(posModel);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        for (Pair<List<String>, List<String>> sent : sentences){
            String[] tokensArr = new String[sent.getSecond().size()];
            tokensArr = sent.getSecond().toArray(tokensArr);
            String[] tags = tagger.tag(tokensArr);
            for (int i = 0; i < tags.length; i++){
                String goldTag = sent.getFirst().get(i);
                String predictedTag = tags[i];
                incrementMap(goldMap, goldTag);
                incrementMap(predictMap, predictedTag);
                if (goldTag.equals(predictedTag)){
                    incrementMap(correctMap, goldTag);
                }
            }
        }
        Map<String, Pair<Double, Double>> result = producePerformance(goldMap, predictMap, correctMap);
        System.out.println("==========OpenNLP Maxent POS PERFORMANCE==========");
        printPerformance(result);
    }

    public static void incrementMap(Map<String, Integer> target, String key){
        if (target.containsKey(key)){
            target.put(key, target.get(key) + 1);
        }
        else {
            target.put(key, 1);
        }
    }

    public static Map<String, Pair<Double, Double>> producePerformance(Map<String, Integer> goldMap, Map<String, Integer> predictMap, Map<String, Integer> correctMap){
        Set<String> labelSet = new HashSet<>(goldMap.keySet());
        Set<String> labelSetPredict = new HashSet<>(predictMap.keySet());
        labelSet.addAll(labelSetPredict);
        int totalLabeled = 0;
        int totalcorrect = 0;
        Map<String, Pair<Double, Double>> ret = new HashMap<>();
        DecimalFormat df = new DecimalFormat("#.##");
        for (String label : labelSet){
            int predict = predictMap.containsKey(label) ? predictMap.get(label) : 0;
            int labeled = goldMap.containsKey(label) ? goldMap.get(label) : 0;
            totalLabeled += labeled;
            int correct = correctMap.containsKey(label) ? correctMap.get(label) : 0;
            totalcorrect += correct;
            double precision = predict > 0 ? (double)correct / (double) predict : 0.0;
            double recall = labeled > 0 ? (double)correct / (double)labeled : 0.0;
            precision = Double.valueOf(df.format(precision * 100.0));
            recall = Double.valueOf(df.format(recall * 100.0));
            ret.put(label, new Pair<>(precision, recall));
        }
        double acc = (double)totalcorrect / (double)totalLabeled;
        acc = Double.valueOf(df.format(acc * 100.0));
        ret.put("ACC", new Pair<>(acc, acc));
        return ret;
    }

    public static void printPerformance(Map<String, Pair<Double, Double>> rMap){
        List<String> labelList = new ArrayList<>(rMap.keySet());
        Collections.sort(labelList);
        Double accuF1 = 0.0;
        DecimalFormat df = new DecimalFormat("#.##");
        for (String label : labelList){
            double precision = rMap.get(label).getFirst();
            double recall = rMap.get(label).getSecond();
            double f1 = 0.0;
            if (precision * recall != 0.0){
                f1 = 2 * precision * recall / (precision + recall);
            }
            f1 = Double.valueOf(df.format(f1));
            System.out.println(label + "\t" + precision + "\t" + recall + "\t" + f1);
            accuF1 += f1;
        }
        System.out.println("Average F1\t" + df.format(accuF1 / (double)labelList.size()));
        System.out.println("Accuracy\t" + rMap.get("ACC").getFirst());
    }

    public static void main(String[] args){
        cogcompNLP();
        CoreNLP();
        OpenNLP_maxent();
    }
}
