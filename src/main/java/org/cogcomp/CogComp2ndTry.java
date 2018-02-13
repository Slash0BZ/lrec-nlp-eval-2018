package org.cogcomp;

import edu.illinois.cs.cogcomp.annotation.AnnotatorException;
import edu.illinois.cs.cogcomp.annotation.AnnotatorService;
import edu.illinois.cs.cogcomp.annotation.TextAnnotationBuilder;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.TextAnnotation;
import edu.illinois.cs.cogcomp.nlp.tokenizer.StatefulTokenizer;
import edu.illinois.cs.cogcomp.nlp.utility.TokenizerTextAnnotationBuilder;
import edu.illinois.cs.cogcomp.pipeline.main.PipelineFactory;
import edu.illinois.cs.cogcomp.pos.POSAnnotator;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

// org.cogcomp.CogComp2ndTry

/**
 * Created by daniel on 1/24/18.
 */
public class CogComp2ndTry {
//    public static String f = "/Users/daniel/Desktop/corpusFilesPlain";
    public static String f = "data/plain";

    public static void main(String[] args) throws IOException, AnnotatorException {
        System.out.println("Starting to execute thins ... ");
        if(args[0].contains("tokenizer")) {
            tokenizer();
        }
        else if(args[0].contains("pos")) {
            pos();
        }
        else if(args[0].contains("lemma")) {
            lemma();
        }
        else if(args[0].contains("chunker")) {
            chunker();
        }
        else if(args[0].contains("ner")) {
            ner();
        }
    }

    public static void tokenizer() throws IOException, AnnotatorException {
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        AnnotatorService pipeline = PipelineFactory.buildPipeline();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".txt")) {
                String content = new String(Files.readAllBytes(Paths.get(listOfFiles[i].getPath())));
                TextAnnotation ta = pipeline.createAnnotatedTextAnnotation( "", "", content);
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
    }

    public static void pos() throws IOException, AnnotatorException {
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        TextAnnotationBuilder stab =
                new TokenizerTextAnnotationBuilder(new StatefulTokenizer());

        POSAnnotator posAnnotator = new POSAnnotator();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".txt")) {
                String content = new String(Files.readAllBytes(Paths.get(listOfFiles[i].getPath())));
                String corpus = "test";
                String textId = Integer.toString(i);
                TextAnnotation ta = stab.createTextAnnotation(corpus, textId, content);
                ta.addView(posAnnotator);
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
    }

    public static void lemma() throws IOException, AnnotatorException {
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        AnnotatorService pipeline = PipelineFactory.buildPipeline(ViewNames.POS, ViewNames.LEMMA);
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".txt")) {
                String content = new String(Files.readAllBytes(Paths.get(listOfFiles[i].getPath())));
                TextAnnotation ta = pipeline.createAnnotatedTextAnnotation("", "", content);
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
    }

    public static void chunker() throws IOException, AnnotatorException {
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        AnnotatorService pipeline = PipelineFactory.buildPipeline(ViewNames.POS, ViewNames.LEMMA,  ViewNames.SHALLOW_PARSE);
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".txt")) {
                String content = new String(Files.readAllBytes(Paths.get(listOfFiles[i].getPath())));
                TextAnnotation ta = pipeline.createAnnotatedTextAnnotation("", "", content);
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
    }

    public static void ner() throws IOException, AnnotatorException {
        File folder = new File(f);
        File[] listOfFiles = folder.listFiles();
        AnnotatorService pipeline = PipelineFactory.buildPipeline(ViewNames.POS, ViewNames.LEMMA, ViewNames.SHALLOW_PARSE, ViewNames.NER_CONLL);
        System.out.println("Starting to evaluate . . . ");
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".txt")) {
                String content = new String(Files.readAllBytes(Paths.get(listOfFiles[i].getPath())));
                TextAnnotation ta = pipeline.createAnnotatedTextAnnotation("", "", content);
            } else if (listOfFiles[i].isDirectory()) {
                System.out.println("Directory " + listOfFiles[i].getName());
            }
        }
    }

}
