/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;

import ca.mcgill.networkdynamics.geoinference.util.Files;

import edu.ucla.sspace.common.ArgOptions;

import edu.ucla.sspace.util.LineReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

import gnu.trove.set.TLongSet;

import gnu.trove.set.hash.TLongHashSet;


/**
 * The main class for running the geoinference methods from the command-line.
 */
public class App {
    
    public static void main(String[] args) {
        if (args.length < 1) {
            mainUsage();
            return;
        }
        
        String command = args[0];
        if (command.equals("train")) 
            train(Arrays.copyOfRange(args, 1, args.length));
        else if (command.equals("cross-validate")) 
            crossValidate(Arrays.copyOfRange(args, 1, args.length));
        else if (command.equals("infer")) 
            infer(Arrays.copyOfRange(args, 1, args.length));
        else {
            throw new IllegalArgumentException(
                "unrecognized command: " + command);
        }            
    }

    private static void mainUsage() {
        System.out.println("usage: java App [command] [args...]\n\n" +
                           "  available commands:\n" +
                           "    train --trains a new geoinference method\n"+
                           "    infer --predicts using a trained model\n"+
                           "    cross-valdiate --trains and tests a model\n"+
                           "  Run any command to see its specific options");
    }

    private static void train(String[] args) {
        ArgOptions opts = createTrainArgOpts(args);

        if (opts.numPositionalArgs() != 3 || opts.hasOption('h')) {
            System.out.println("usage: java App train [options] GeoMethodClass "
                               + "dataset-dir/ output-dir/\n\n" +
                               opts.prettyPrint());
            return;
        }

        String geoinfClassName = opts.getPositionalArg(0);
        File datasetDir = new File(opts.getPositionalArg(1));
        File modelDir = new File(opts.getPositionalArg(2));

        if (!datasetDir.exists()) {
            throw new IllegalArgumentException(
                "Provided dataset directory does not exist:" + datasetDir);
        }

        if (modelDir.exists() && !opts.hasOption('f')) {
            throw new IllegalArgumentException(
                "Provided model directory already exist!:" + modelDir
                +  " .  Either specify a non-existing directory or use " 
                + "-f to overwrite the current directory's contents.");
        }
        if (!modelDir.exists())
            modelDir.mkdir();

        JSONObject settings = loadSettings(opts);

        Dataset dataset = opts.hasOption('S')
            ? new SparseFormatDataset(datasetDir)
            : new DenseFormatDataset(datasetDir);


        // Instantiate the method
        GeoinferenceMethod method = instantiate(geoinfClassName);         

        // Train the model, leaving it to the model's code to save the trained
        // system in the specified modelDir
        GeoinferenceModel trainedModel = 
            method.train(settings, dataset, modelDir);
    }

    private static ArgOptions createTrainArgOpts(String args[]) {
        ArgOptions opts = new ArgOptions();
        
        opts.addOption('h', "help", "Generates a help message and exits",
                          false, null, "Program Options");

        opts.addOption('f', "force", "Overwrite the existing output",
                          false, null, "Program Options");

        opts.addOption('l', "location-source", "Specifies the source of "+
                       "ground-truth locations",
                       true, "LocType", "Program Options");

        opts.addOption('S', "use-sparse-input", 
                       "Indicates the training can be performed using " +
                       "the sparse input format",
                       false, null, "Program Options");

        opts.addOption('s', "settings", "An optional settings file that is " +
                       "passed to the geoinference method",
                       true, "JSON_FILE", "Program Options");

        opts.parseOptions(args);
        return opts;
    }

    /**
     *
     */
    private static void crossValidate(String[] args) {
        ArgOptions opts = createCvArgOpts(args);

        if (opts.numPositionalArgs() != 4 || opts.hasOption('h')) {
            System.out.println("usage: java App cross-validate [options] " +
                               "GeoMethodClass dataset-dir/ folds-dir/ " +
                               "results-dir/\n\n" + opts.prettyPrint());
            return;
        }

        String geoinfClassName = opts.getPositionalArg(0);
        File datasetDir = new File(opts.getPositionalArg(1));
        File foldsDir = new File(opts.getPositionalArg(2));
        File resultsDir = new File(opts.getPositionalArg(3));

        if (!datasetDir.exists()) {
            throw new IllegalArgumentException(
                "Provided dataset directory does not exist:" + datasetDir);
        }

        if (!foldsDir.exists()) {
            throw new IllegalArgumentException(
                "Provided folds directory does not exist:" + foldsDir);
        }

        File foldsInfoFile = new File(foldsDir, "folds.info.tsv");
        if (!foldsInfoFile.exists()) {
            throw new IllegalArgumentException(
                "Provided folds directory does not contain its " +
                "self-describing file:" + foldsInfoFile);
        }

        if (resultsDir.exists() && !opts.hasOption('f')) {
            throw new IllegalArgumentException(
                "Provided results directory already exist!:" + resultsDir
                +  " .  Either specify a non-existing directory or use " 
                + "-f to overwrite the current directory's contents.");
        }
        if (!resultsDir.exists())
            resultsDir.mkdir();



        // Load the setting from file, if it exists
        JSONObject settings = loadSettings(opts);

        // Instantiate the method
        GeoinferenceMethod method = instantiate(geoinfClassName);      

        List<String[]> folds = new ArrayList<String[]>();
        for (String line : new LineReader(foldsInfoFile)) {
            String[] cols = line.split("\t");
            folds.add(cols);
        }

        for (String[] fold : folds) {                              
            String foldName = fold[0];
            // These are the posts that need to be held out, but we filter at
            // the user level, so this file is not used
            File heldOutPostIdsFile = new File(foldsDir, fold[1]);
            // This is the file with the user IDs that need to be held-out
            File heldOutUserIdsFile = new File(foldsDir, fold[2]);
            // These is the gold standard users used for prediction
            File usersFileTestSet = new File(foldsDir, fold[3]);

            // If the user requested a specific fold to be run, skip all others.
            if (opts.hasOption('F') 
                    && !opts.getStringOption('F').equals(foldName)) {
                continue;
            }

            // Load in the users that will be excluded for testing purposes
            TLongSet usersToExclude = new TLongHashSet();
            for (String line : new LineReader(heldOutUserIdsFile))
                usersToExclude.add(Long.parseLong(line));
            
            Dataset dataset = opts.hasOption('S')
                ? new SparseFormatDataset(datasetDir, usersToExclude)
                : new DenseFormatDataset(datasetDir, usersToExclude);


            System.out.printf("Training %s on fold %s%n", 
                              geoinfClassName, foldName);
            GeoinferenceModel model = method.train(settings, dataset, null);
            
            // Use dummy variables for location type since we only need this
            // class to read the users file data only.
            Dataset testingData = new DenseFormatDataset(
                foldsDir, "", new TLongHashSet(), usersFileTestSet.getName());

            PrintWriter foldOutput = null;
            try {
                foldOutput = Files.openGzWrite(
                    new File(resultsDir, foldName + ".results.tsv.gz"));
            } catch (IOException ioe) {
                throw new IllegalStateException("Unable to create output file "
                                                + foldName + ".results.tsv.gz");
            }

            // For general bookkeeping
            int numUsersSeen = 0;
            int numPostsSeen = 0;

            for (Map.Entry<Long,List<JSONObject>> userIdAndPosts 
                     : testingData.getPostsByUser()) {
               
                long userId = userIdAndPosts.getKey();
                List<JSONObject> posts = userIdAndPosts.getValue();
                List<LatLon> postLocations = 
                    model.inferPostLocationsByUser(userId, posts);
                
                if (posts.size() != postLocations.size()) {
                    throw new IllegalStateException(
                        "Model returned an unexpected number of predictions. "
                        + "Got " + postLocations.size() 
                        + ", expected: " + posts.size());
                }

                for (int i = 0; i < posts.size(); ++i) {
                    try {
                        long postId = posts.get(i).getLong("id");
                        LatLon loc = postLocations.get(i);
                        if (loc != null)
                            foldOutput.println(postId +"\t" + loc.lat +"\t"+ loc.lon);
                    } catch (JSONException je) {
                        throw new IllegalStateException("no post id", je);
                    }
                }
                numUsersSeen++;
                numPostsSeen += posts.size();
                if (numUsersSeen % 10_000 == 0) {
                    System.out.printf("During testing of fold %s, processed %d"+
                                      " users and %d posts%n",
                                      foldName, numUsersSeen, numPostsSeen);
                }
            }

            foldOutput.close();
        }
    }

    private static ArgOptions createCvArgOpts(String args[]) {
        ArgOptions opts = new ArgOptions();
        
        opts.addOption('h', "help", "Generates a help message and exits",
                          false, null, "Program Options");

        opts.addOption('f', "force", "Overwrite the existing output",
                          false, null, "Program Options");

        opts.addOption('l', "location-source", "Specifies the source of "+
                       "ground-truth locations",
                       true, "LocType", "Program Options");
        opts.addOption('s', "settings", "An optional settings file that is " +
                       "passed to the geoinference method",
                       true, "JSON_FILE", "Program Options");

        opts.addOption('S', "use-sparse-input", 
                       "Indicates the training can be performed using " +
                       "the sparse input format",
                       false, null, "Program Options");

        opts.addOption('F', "fold", "Specifies a specific fold to run " +
                       "from the dataset",
                       true, "String", "Program Options");

        opts.parseOptions(args);
        return opts;
    }

    
    private static void infer(String[] args) {
        ArgOptions opts = createInferArgOpts(args);

        if (opts.numPositionalArgs() != 4 || opts.hasOption('h')) {
            System.out.println("usage: java App cross-validate [options] " +
                               "GeoMethodClass model-dir/ " +
                               "dataset-to-classify-dir/ " +
                               "output-inferences.tsv.gz\n\n" +
                               opts.prettyPrint());
            return;
        }

        String geoinfClassName = opts.getPositionalArg(0);
        File modelDir = new File(opts.getPositionalArg(1));
        File datasetDir = new File(opts.getPositionalArg(2));
        File inferencesFile = new File(opts.getPositionalArg(3));

        if (!datasetDir.exists()) {
            throw new IllegalArgumentException(
                "Provided dataset directory does not exist:" + datasetDir);
        }

        if (!modelDir.exists()) {
            throw new IllegalArgumentException(
                "Provided model directory does not exist:" + modelDir);
        }

        if (inferencesFile.exists() && !opts.hasOption('f')) {
            throw new IllegalArgumentException(
                "Provided inferences file already exist!:" + inferencesFile
                +  " .  Either specify a non-existing file or use " 
                + "-f to overwrite the current file's contents.");
        }

        // Load the settings, if any, used to configure the trained model's
        // behavior
        JSONObject settings = loadSettings(opts);

        // Instantiate the method that was used for training
        GeoinferenceMethod method = instantiate(geoinfClassName); 

        // Now load the trained model according to the system's logic
        GeoinferenceModel model = method.load(settings, modelDir);

        boolean inferByUser = true;
        if (opts.hasOption('i')) {
            String byType = opts.getStringOption('i');
            if (byType.equals("post"))
                inferByUser = false;
            else if (!byType.equals("user")) {
                throw new IllegalArgumentException(
                    "Unknown inference type: " +byType 
                    +"; expected either 'post' or 'user'");
                                                   
            }
        }

        Dataset dataset = opts.hasOption('S')
            ? new SparseFormatDataset(datasetDir)
            : new DenseFormatDataset(datasetDir);

        PrintWriter inferOutput = null;
        try {
            inferOutput = (inferencesFile.getName().endsWith(".gz"))
                ? Files.openGzWrite(inferencesFile)
                : new PrintWriter(inferencesFile);
        } catch (IOException ioe) {
            throw new IllegalStateException("Unable to create output file "
                                            + inferencesFile);
        }
        
        // For general bookkeeping
        int numUsersSeen = 0;
        int numPostsSeen = 0;
        
        for (Map.Entry<Long,List<JSONObject>> userIdAndPosts 
                 : dataset.getPostsByUser()) {
               
            long userId = userIdAndPosts.getKey();
            List<JSONObject> posts = userIdAndPosts.getValue();
            List<LatLon> postLocations = null;            

            if (inferByUser) {
                postLocations = 
                    model.inferPostLocationsByUser(userId, posts);
            }
            else {
                postLocations = new ArrayList<LatLon>(posts.size());
                for (JSONObject post : posts)
                    postLocations.add(model.inferLocation(post));
            }
            
            if (posts.size() != postLocations.size()) {
                throw new IllegalStateException(
                    "Model returned an unexpected number of predictions. "
                    + "Got " + postLocations.size() 
                    + ", expected: " + posts.size());
            }

            for (int i = 0; i < posts.size(); ++i) {
                try {
                    long postId = posts.get(i).getLong("id");
                    LatLon loc = postLocations.get(i);
                    if (loc != null)
                        inferOutput.println(postId +"\t" + loc.lat +"\t"+ loc.lon);
                } catch (JSONException je) {
                    throw new IllegalStateException("no post id", je);
                }
            }
            numUsersSeen++;
            numPostsSeen += posts.size();
            if (numUsersSeen % 10_000 == 0) {
                System.out.printf("During inference, processed %d"+
                                  " users and %d posts%n",
                                  numUsersSeen, numPostsSeen);
            }
        }
        
        inferOutput.close();

    }

    private static ArgOptions createInferArgOpts(String args[]) {
        ArgOptions opts = new ArgOptions();
        
        opts.addOption('h', "help", "Generates a help message and exits",
                          false, null, "Program Options");
        opts.addOption('f', "force", "Overwrite the existing output",
                          false, null, "Program Options");
        opts.addOption('s', "settings", "An optional settings file that is " +
                       "passed to the geoinference method",
                       true, "JSON_FILE", "Program Options");
        opts.addOption('S', "use-sparse-input", 
                       "Indicates the training can be performed using " +
                       "the sparse input format",
                       false, null, "Program Options");

        opts.addOption('i', "infer-by", "Specifies how inferences should " +
                       "be performed (Default: user)", true, "user|post",
                       "Program Options");

        opts.parseOptions(args);
        return opts;
    }

    private static JSONObject loadSettings(ArgOptions opts) {
        // Load the setting from file, if it exists
        JSONObject settings = new JSONObject();
        if (opts.hasOption('s')) {
            StringBuilder sb = new StringBuilder();
            File settingsFile = new File(opts.getStringOption('s'));
            if (!settingsFile.exists()) {
                throw new IllegalArgumentException(
                    "Provided settings .json does not exist:" + settingsFile);
            }

            // Slurp the JSON file's contents
            try {
                settings = new JSONObject(new JSONTokener(
                    new BufferedReader(new FileReader(settingsFile))));
            } 
            catch (FileNotFoundException | JSONException ex) {
                throw new IllegalStateException(
                    "Settings JSON is not valid", ex);
            }
        }
        return settings;
    }

    private static GeoinferenceMethod instantiate(String geoinfClassName) {
        // Instantiate the method
        GeoinferenceMethod method = null;
        try {
            Object o = Class.forName(geoinfClassName).newInstance();
            if (!(o instanceof GeoinferenceMethod)) {
                throw new IllegalStateException(
                    "Provided class does not implement GeoinferenceMethod:"
                    + geoinfClassName);
            }
            method = (GeoinferenceMethod)o;
        } catch (InstantiationException | ClassNotFoundException 
                 | IllegalAccessException ex) {
                throw new IllegalStateException(
                    "Provided class cannot be instantiated:"
                    + geoinfClassName, ex);            
        }
        return method;
    }
}
