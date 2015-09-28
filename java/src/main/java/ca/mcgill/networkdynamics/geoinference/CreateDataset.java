/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.util.regex.*;
import java.util.stream.*;
import java.util.zip.*;
import gnu.trove.iterator.*;
import gnu.trove.map.*;
import gnu.trove.map.hash.*;
import gnu.trove.set.*;
import gnu.trove.set.hash.*;
import edu.ucla.sspace.common.ArgOptions;
import edu.ucla.sspace.util.*;
import org.json.*;
import com.google.common.cache.*;
import com.google.common.util.concurrent.*;

/** 
 * The command-line executable class that generates a dataset from one or more
 * files containing Twitter JSON data.  A dataset is a directory containing
 * precomputed files for the geoinference methods to use directly, such as the
 * social network or estimated locations from GPS data.
 */
public class CreateDataset {

    static final AtomicInteger fileNo = new AtomicInteger();
    
    static final int MAX_USERS_PER_TMP_FILE = 10_000;

    static final int MAX_OPEN_FILES = 512;
    
    // set after createUsersFile
    static long totalNumPosts = 0;
    
    public static void main(String[] args) throws Exception {

        ArgOptions opts = createArgOpts(args);

        if (opts.numPositionalArgs() < 2 || opts.hasOption('h')) {
            System.out.println("usage: java CreateDataset [options] "
                               + "output-dataset-dir/ input-files...\n\n" +
                               opts.prettyPrint());
            return;
        }

        File datasetDir = new File(opts.getPositionalArg(0));
        int nArgs = opts.numPositionalArgs();

        // Process the input files to see how many we need to process
        List<File> inputFiles = new ArrayList<File>();        
        for (int i = 1; i < nArgs; ++i) 
            loadInputFiles(new File(opts.getPositionalArg(i)), inputFiles);
        System.out.printf("Loaded %d input files for processing%n", inputFiles.size());
        
        // Check for whether the dataset exists and if so, whether we can
        // overwrite it
        boolean canOverwrite = opts.hasOption('f');
        if (datasetDir.exists() && datasetDir.isDirectory()) {
            File usersFile = new File(datasetDir, "users.json.gz");
            File postsFile = new File(datasetDir, "posts.json.gz");
            if (canOverwrite && (postsFile.exists() || usersFile.exists())) {
                System.out.printf("Found prior dataset in %s, but cannot " +
                                  "overwrite it.  Use -f to force overwriting "+
                                  "prior data%n", datasetDir);
                return;
            }
        }
        else {
            datasetDir.mkdir();
        }

        boolean shouldCreateSparseFiles = opts.getBooleanOption('s', true);

        File tmpDir = (opts.hasOption('T'))
            ? new File(opts.getStringOption('T'))
            : new File(System.getProperty("java.io.tmpdir"));
        if (!tmpDir.exists() || !tmpDir.isDirectory()) {
            throw new IllegalStateException(
                tmpDir + " is not a valid temporary directory");
        }
        
        File cvDir = (opts.hasOption('c')) 
            ? new File(opts.getStringOption('c')) : null;
        int numFolds = opts.getIntOption('F', 5); 
        if (cvDir != null && !cvDir.exists())
            cvDir.mkdir();

        /*
         *
         * THE CODE BELOW ACTUALLY CREATES THE DATASET
         *
         */
        
        try {

            boolean shouldCreatePostsInMemory = opts.getBooleanOption('P', false);
            List<File> postFiles = (shouldCreatePostsInMemory)
                ? createUsersFileInMem(inputFiles)
                : createUsersFile(inputFiles, tmpDir, datasetDir);
            if (shouldCreateSparseFiles) 
                createSparseUsersFile(datasetDir);
            TLongSet usersInNetwork = extractMentionNetworks(datasetDir);
            filterUsersByNetwork(datasetDir, usersInNetwork);
            if (cvDir != null)
                createCrossValidationSplits(datasetDir, cvDir, numFolds);

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }
    
    private static List<File> createUsersFileInMem(List<File> inputFiles) {
        throw new UnsupportedOperationException(
            "Sorting posts in memory is currently unsupported");
    }

    private static List<File> createUsersFile(List<File> inputFiles, 
                                              File tmpDir,
                                              File datasetDir) throws IOException {
        TLongObjectMap<File> userToFile =
            new TLongObjectHashMap<File>();

        AtomicInteger curWriterCount = new AtomicInteger();
        AtomicInteger numUsersInCurWriter = new AtomicInteger();
        AtomicReference<File> curWriter = new AtomicReference<File>();


        AtomicLong postsSeen = new AtomicLong();


        AtomicInteger filesProcessed = new AtomicInteger();
        final ReentrantReadWriteLock rwl = new ReentrantReadWriteLock();

        LoadingCache<File,PrintWriter> fileWriters = 
            CacheBuilder.newBuilder()
            .maximumSize(MAX_OPEN_FILES)
            .removalListener(new RemovalListener<File,PrintWriter>() {
                    public void onRemoval(RemovalNotification<File,PrintWriter> notification) {
                                     notification.getValue().close();
                                 }
                })
            .build(new CacheLoader<File, PrintWriter>() {
                     public PrintWriter load(File key) throws Exception {
                         return openGzAppend(key);
                     }
                });

        inputFiles.parallelStream()
            .forEach(f -> {
                    try {
                        BufferedReader br = openGz(f);
                        for (String line = null; (line = br.readLine()) != null; ) {
                            JSONObject post = new JSONObject(line);
                            if (!post.has("user"))
                                continue;
                            JSONObject user = post.getJSONObject("user");
                            long uid = user.getLong("id");
                            File tmpFile = null;
                            rwl.readLock().lock();
                            tmpFile = userToFile.get(uid);
                            rwl.readLock().unlock();                            
                            
                            if (tmpFile == null) {
                                // double lock check
                                rwl.readLock().lock();
                                tmpFile = userToFile.get(uid);
                                rwl.readLock().unlock();                            
                                
                                if (tmpFile == null) {
                                    rwl.writeLock().lock();
                                    if (curWriter.get() == null 
                                        || numUsersInCurWriter.get() == MAX_USERS_PER_TMP_FILE) {
                                        
                                        tmpFile = new File(tmpDir, "users." + 
                                                           curWriterCount + ".json.tmp.gz");
                                        curWriter.set(tmpFile);
                                        curWriterCount.incrementAndGet();
                                        numUsersInCurWriter.set(0);
                                    }
                                    
                                    tmpFile = curWriter.get();
                                    userToFile.put(uid, tmpFile);
                                    numUsersInCurWriter.incrementAndGet();
                                    fileWriters.get(tmpFile).println(line);
                                    rwl.writeLock().unlock();
                                }
                                else {
                                    fileWriters.get(tmpFile).println(line);
                                }
                            }
                            // If the user already has a writer, print its line to this file
                            else {
                                fileWriters.get(tmpFile).println(line);
                            }
                            
                            long ps = postsSeen.incrementAndGet();
                            if (ps % 500_000 == 0) {
                                System.out.printf("Processed %d posts from %d/%d " +
                                                  "input files into %d tmp files%n", 
                                                  ps, filesProcessed.get(), 
                                                  inputFiles.size(), curWriterCount.get());
                            }
                        }
                        br.close();
                        filesProcessed.incrementAndGet();
                    } catch (JSONException je) {
                        je.printStackTrace();
                    } catch (ExecutionException ee) {
                        ee.printStackTrace();
                    } catch (IOException ie) {
                        ie.printStackTrace();
                    }
                });
            

    
        
        System.out.printf("FINISHED processing %d posts from %d/%d " +
                          "input files into %d tmp files%n", 
                          postsSeen.get(), filesProcessed.get(), 
                          inputFiles.size(), curWriterCount.get() +1);

        totalNumPosts = postsSeen.get();


        // Finish writing
        for (PrintWriter pw : fileWriters.asMap().values())
            pw.close();

        File usersFile = new File(datasetDir, "users.json.gz");
        PrintWriter usersPw_ = null;
        try {
            usersPw_ = openGzWrite(usersFile);
        } catch (IOException ioe) {
            throw new IOError(ioe);
        }
        PrintWriter usersPw = usersPw_;

        int numFilesProcessed = 0;
        AtomicLong numPostsRecorded = new AtomicLong();

        Set<File> tmpFiles = new HashSet<File>(userToFile.valueCollection());

        // Sort each file to organize its users
        for (File tmpFile : tmpFiles) {
            
            // TODO(?): check that the file isn't too big to process in memory
            if (isTooBigToProcessInMemory(tmpFile)) {
                throw new UnsupportedOperationException("please file a ticket in github");
            }
            
            List<JSONObject> sortedPosts = new ArrayList<JSONObject>(1_000_000);
            BufferedReader br = openGz(tmpFile);
            for (String line = null; (line = br.readLine()) != null; ) {
                try {
                    sortedPosts.add(new JSONObject(line));
                } catch (JSONException je) {
                    je.printStackTrace();
                }
            }
            br.close();

            // Clean up our mess, since this file is no longer needed
            tmpFile.delete();

            Collections.sort(sortedPosts, new Comparator<JSONObject>() {
                    public int compare(JSONObject jo1, JSONObject jo2) {
                        try {
                            long uid1 = jo1.getJSONObject("user").getLong("id");
                            long uid2 = jo2.getJSONObject("user").getLong("id");
                            return (uid1 == uid2) 
                                ? Long.compare(jo1.getLong("id"), jo2.getLong("id"))
                                : Long.compare(uid1, uid2);
                        } catch (JSONException je) {
                            throw new IllegalStateException(je);
                        }
                    }
                });


            sortedPosts.parallelStream()
                .collect(Collectors.groupingBy(post -> getUid(post)))
                .entrySet().forEach(e -> {
                        long uid = e.getKey();
                        List<JSONObject> posts = e.getValue();
                        try {
                            JSONObject combined = new JSONObject();
                            combined.put("user_id", uid);
                            JSONArray postsArr = new JSONArray();
                            for (JSONObject post : posts)
                            postsArr.put(post);
                            combined.put("posts", postsArr);
                            synchronized(usersPw) {
                                usersPw.println(combined);
                            }

                            numPostsRecorded.addAndGet(posts.size());
                        } catch (JSONException je) {
                            throw new IllegalStateException(je);
                        }
                    });
            
            ++numFilesProcessed;            

            System.out.printf("Processed tmp file %d/%d, recorded %d posts%n",
                              numFilesProcessed, tmpFiles.size(),
                              numPostsRecorded.get());
        }

        usersPw.close();
        System.out.println("Done creating users file");
        return null;
    }


    private static void createSparseUsersFile(File datasetDir) throws Exception {
        File usersFile = new File(datasetDir, "users.json.gz");
        File sparseUsersFile = new File(datasetDir, "users.tsv.gz");

        PrintWriter pw = openGzWrite(sparseUsersFile);

        Pattern WHITESPACE = Pattern.compile("[\\p{Space}]+", Pattern.UNICODE_CHARACTER_CLASS);
        Matcher m = WHITESPACE.matcher("");
        StringBuilder sb = new StringBuilder(10_000);

        long numUsers = 0, numPosts = 0;

        BufferedReader br = openGz(usersFile);
        for (String line = null; (line = br.readLine()) != null; ) {
            try {
                JSONObject jo = new JSONObject(line);
                long uid = jo.getLong("user_id");
                JSONArray postsArr = jo.getJSONArray("posts");
                int n = postsArr.length();
                sb.setLength(0);
                for (int i = 0; i < n; ++i) {

                    JSONObject post = postsArr.getJSONObject(i);
                    String text = m.reset(post.getString("text"))
                        .replaceAll(" ").trim();

                    JSONObject user = post.getJSONObject("user");
                    String postId = post.getString("id_str");
                    String userId = user.getString("id_str");
                    String userLoc = m.reset(user.getString("location"))
                        .replaceAll(" ").trim();
                    
                    String createdAt = post.getString("created_at");

                    if (sb.length() == 0)
                        sb.append(userId);

                    sb.append('\t').append(text)
                        .append('\t').append(postId)
                        .append('\t').append(userLoc);
                    getGeo(post, sb);
                    getMentions(post, sb);
                    getHashTags(post, sb);
                    getIsRetweet(post, sb);
                    getPlace(post, sb);
                    
                }

                pw.println(sb);
                
                numUsers++;
                numPosts += n;

                if (numUsers % 100_000 == 0) {
                    System.out.printf("Sparsifying data: processed %d users " +
                                      "and %d/%d posts (%.2f done)%n",
                                      numUsers, numPosts, totalNumPosts, 
                                      (double)numPosts / totalNumPosts);
                }
            } catch (JSONException je) {
                je.printStackTrace();
            }
        }
        System.out.println("Done sparsifying users file");
        pw.close();
    }

    static void getGeo(JSONObject post, StringBuilder sb) throws Exception {
        sb.append('\t');
        if (!post.has("geo") || post.isNull("geo")) 
            return;
        JSONObject geo = post.getJSONObject("geo");
        JSONArray coordinates =  geo.getJSONArray("coordinates");
        sb.append(coordinates.getDouble(0))
            .append(' ')
            .append(coordinates.getDouble(1));
    }

    static void getMentions(JSONObject post, StringBuilder sb) throws Exception {
        sb.append('\t');
        if (!post.has("entities")) 
            return;
        JSONObject entities = post.getJSONObject("entities");
        JSONArray mentions = entities.getJSONArray("user_mentions");
        int n = mentions.length();
        if (n == 0)
            return;
        for (int i = 0; i < n; ++i) {
            JSONObject mention = mentions.getJSONObject(i);
            if (mention.has("id_str") && !mention.isNull("id_str")) {
                String uid = mention.getString("id_str");
                if (i > 0)
                    sb.append(' ');
                sb.append(uid);
            }
        }        
    }

    static void getHashTags(JSONObject post, StringBuilder sb) throws Exception {
        sb.append('\t');
        if (!post.has("entities")) 
            return;
        JSONObject entities = post.getJSONObject("entities");
        JSONArray hashtags = entities.getJSONArray("hashtags");
        int n = hashtags.length();
        if (n == 0)
            return;
        for (int i = 0; i < n; ++i) {
            if (i > 0)
                sb.append(' ');
            sb.append(hashtags.getJSONObject(i).getString("text"));
        }
    }

    static void getIsRetweet(JSONObject post, StringBuilder sb) throws Exception {
        sb.append('\t');
        if (post.has("retweeted_status"))
            sb.append("True");
        else
            sb.append("False");
    }

    static void getPlace(JSONObject post, StringBuilder sb) throws Exception {
        sb.append('\t');
        if (post.has("place") && !post.isNull("place")) {
            sb.append(post.getJSONObject("place"));
        }
    }

    private static long getUid(JSONObject post) {
        try {
            return post.getJSONObject("user").getLong("id");
        } catch (JSONException je) {
            throw new IllegalStateException(je);
        }
    }

    private static boolean isTooBigToProcessInMemory(File tmpFile) {
        return false;
    }

    private static TLongSet extractMentionNetworks(File datasetDir) throws IOException {

        TLongObjectMap<TLongSet> userToMentioned = 
            new TLongObjectHashMap<TLongSet>(50_000_000);

        // Read from the sparse dataset since it should be faster
        File sparseUsersFile = new File(datasetDir, "users.tsv.gz");        
        BufferedReader br = openGz(sparseUsersFile);
        long numLines = 0;

        // In the first pass, just keep track of reciprocal mentions and ignore
        // weights and time
        for (String line = null; (line = br.readLine()) != null; ) {
            
            String[] arr = line.split("\t", -1);
            // System.out.printf("Saw %d columns%n", arr.length);           
            if ((arr.length-1) % 8 != 0) {
                throw new IllegalStateException(
                    "Sparse-formatted user.tsv line has too " +
                                 "few columns");
            }

            long userId = Long.parseLong(arr[0]);                   
                    
            for (int i = 1; i < arr.length; i += 8) {

                String mentionsStr = arr[i+4];
                for (String um : mentionsStr.split(" ")) {
                    if (um.length() == 0) 
                        break;
                    long mentioned = Long.parseLong(um);
                    TLongSet sinks = userToMentioned.get(userId);
                    if (sinks == null) {
                        sinks = new TLongHashSet();
                        userToMentioned.put(userId, sinks);
                    }
                    sinks.add(mentioned);
                }
            }
            if (++numLines % 1_000_000 == 0) {
                System.out.printf("Building network, processed %d lines, " +
                                  "saw %d users mentioning%n",
                                  numLines, userToMentioned.size());
            }
        }
        br.close();

        System.out.println("Done reading in data; finding network reciprocity");

        PrintWriter diGraphPw = 
            new PrintWriter(new File(datasetDir, "mention_network.directed.elist"));
        PrintWriter recipGraphPw = 
            new PrintWriter(new File(datasetDir, "bi_mention_network.elist"));

        // We create this map using the reciprocal information to avoid having a
        // lot of unnecessary bookkeeping in memory for non-reciprocal mentions
        TLongObjectMap<TLongIntMap> reciprocalUserToWeighedMentioned = 
            new TLongObjectHashMap<TLongIntMap>(50_000_000);


        TLongObjectIterator<TLongSet> iter = userToMentioned.iterator();
        while (iter.hasNext()) {
            iter.advance();
            long from = iter.key();
            TLongSet mentioned = iter.value();
            TLongIterator iter2 = mentioned.iterator();
            while (iter2.hasNext()) {
                long to = iter2.next();

                TLongSet reverse = userToMentioned.get(to);
                boolean isReciprocal = reverse != null && reverse.contains(from);

                diGraphPw.println(from + " " + to);

                // Check for bidirectionality and the use the ID ordering to avoid
                // double printing the edge               
                if (isReciprocal) {
                    if (to < from)
                        recipGraphPw.println(from + " " + to);

                    // Initialze the weighted graph for the next pass
                    TLongIntMap m = reciprocalUserToWeighedMentioned.get(from);
                    if (m == null) {
                        m = new TLongIntHashMap();
                        reciprocalUserToWeighedMentioned.put(from, m);
                    }
                    m.put(to, 0); 

                    m = reciprocalUserToWeighedMentioned.get(to);
                    if (m == null) {
                        m = new TLongIntHashMap();
                        reciprocalUserToWeighedMentioned.put(to, m);
                    }
                    m.put(from, 0); 
                    
                }
            }
        }

        userToMentioned = null; // For GC
        System.out.println("Done creating reciprocal and fully-directed networks");

        diGraphPw.close();
        recipGraphPw.close();

        br = openGz(sparseUsersFile);
        numLines = 0;

        // Second pass, keep track of weights
        for (String line = null; (line = br.readLine()) != null; ) {

            String[] arr = line.split("\t", -1);
            long userId = Long.parseLong(arr[0]);                   
            TLongIntMap mentionedToWeight = 
                reciprocalUserToWeighedMentioned.get(userId);
            
            // A null value indicates this user does not participate in
            // reciprocal mentions
            if (mentionedToWeight == null)
                continue;
                    
            for (int i = 1; i < arr.length; i += 8) {
                String mentionsStr = arr[i+4];
                for (String um : mentionsStr.split(" ")) {
                    if (um.length() == 0) 
                        break;

                    long mentioned = Long.parseLong(um);
                    
                    // Does not update if mention wasn't in the map, which it
                    // should be after the initializeation step
                    mentionedToWeight.adjustValue(mentioned, 1);
                }
            }
            if (++numLines % 1_000_000 == 0) {
                System.out.printf("Rescanning network for weights, " +
                                  "processed %d lines", numLines);
            }
        }
        br.close();

        PrintWriter weighedDiGraphPw = 
            new PrintWriter(new File(datasetDir, "bi_mention_network.directed.weighted.elist"));
        PrintWriter recipWeightedGraphPw = 
            new PrintWriter(new File(datasetDir, "bi_mention_network.weighted.elist"));

        TLongObjectIterator<TLongIntMap> witer = 
            reciprocalUserToWeighedMentioned.iterator();
        while (witer.hasNext()) {
            witer.advance();
            long from = witer.key();
            TLongIntMap mentionToWeighted = witer.value();
            // TLongIterator iter2 = mentioned.iterator();
            TLongIntIterator witer2 = mentionToWeighted.iterator();


            while (witer2.hasNext()) {
                witer2.advance();
                long to = witer2.key();
                int freq = witer2.value();               
                
                weighedDiGraphPw.println(from + " " + to + " " + freq);

                if (to < from) {
                    int revFreq = reciprocalUserToWeighedMentioned
                        .get(to).get(from);
                    recipWeightedGraphPw.println(from + " " + to + " " + 
                                                 Math.min(freq, revFreq));
                }
            }
        }

        weighedDiGraphPw.close();
        recipWeightedGraphPw.close();

        System.out.println("Done creating weighted and directed networks");

        return new TLongHashSet(reciprocalUserToWeighedMentioned.keySet());
    }
                
    private static void filterUsersByNetwork(File datasetDir, 
                                             TLongSet usersInNetwork) throws IOException {
        File usersFile = new File(datasetDir, "users.json.gz");
        File filteredUsersFile = new File(datasetDir, "users.only-in-network.json.gz");
        
        PrintWriter pw = openGzWrite(filteredUsersFile);
        BufferedReader br = openGz(usersFile);
        for (String line = null; (line = br.readLine()) != null; ) {
            try {
                JSONObject jo = new JSONObject(line);
                long uid = jo.getLong("user_id");
                if (usersInNetwork.contains(uid))
                    pw.println(line);
            } catch (JSONException je) {
                je.printStackTrace();
            }
        }
        br.close();        
        pw.close();

        File sparseUsersFile = new File(datasetDir, "users.tsv.gz");
        if (!sparseUsersFile.exists())
            return;

        File filteredSparseUsersFile = new File(datasetDir, "users.only-in-network.tsv.gz");
        pw = openGzWrite(filteredSparseUsersFile);
        br = openGz(sparseUsersFile);

        for (String line = null; (line = br.readLine()) != null; ) {            
            int i = line.indexOf('\t');
            if (i < 0)
                continue;

            long userId = Long.parseLong(line.substring(0, i));
            pw.println(line);
        }

        pw.close();
        br.close();
    }

    private static void createCrossValidationSplits(File datasetDir, 
                                                    File cvDir,
                                                    int numFolds) throws IOException {
        if (numFolds < 2) {
            throw new IllegalArgumentException();
        }

        PrintWriter[] postIdPws = new PrintWriter[numFolds];
        PrintWriter[] userIdPws = new PrintWriter[numFolds];
        PrintWriter[] goldLocPws = new PrintWriter[numFolds];
        PrintWriter[] testDataPws = new PrintWriter[numFolds];
        PrintWriter foldInfoPw = 
            new PrintWriter(new File(cvDir, "folds.info.tsv"));

        // Initialize the cross-validation writers
        for (int i = 0; i < numFolds; ++i) {
            String foldName = "fold_" + i;
            postIdPws[i] = new PrintWriter(new File(cvDir, foldName + ".post-ids.txt"));
            userIdPws[i] = new PrintWriter(new File(cvDir, foldName + ".user-ids.txt"));
            goldLocPws[i] = new PrintWriter(new File(cvDir, foldName + ".gold-locations.tsv"));
            testDataPws[i] = new PrintWriter(new File(cvDir, foldName + ".users.json.gz"));
            foldInfoPw.printf("%s\t%s.post-ids.txt\t%s.user-ids.txt\t" +
                              "%s.users.json.gz\n",
                              foldName, foldName, foldName, foldName);
        }
        foldInfoPw.close();

        System.out.printf("Creating %d cross-validation folds in %s%n",
                          numFolds, cvDir);

        File usersFile = new File(datasetDir, "users.json.gz");
        BufferedReader br = openGz(usersFile);
        long numUsers = 0, numPosts = 0, numGoldPosts = 0, numGoldUsers = 0;

        for (String line = null; (line = br.readLine()) != null; ) {
            try {
                JSONObject jo = new JSONObject(line);
                long uid = jo.getLong("user_id");
                JSONArray postsArr = jo.getJSONArray("posts");
                int n = postsArr.length();

                List<JSONObject> goldPosts = new ArrayList<JSONObject>();

                // Look through this user's posts to pull out any geo-tagged posts
                for (int i = 0; i < n; ++i) {
                    JSONObject post = postsArr.getJSONObject(i);
                    if (post.has("geo") && !post.isNull("geo")) 
                        goldPosts.add(post);                    
                }

                numPosts += n;
                numUsers++;

		if (numUsers % 100000 == 0) {
                    System.out.printf("Processed %d users, saw %d gold so far "
                                      +"(%d posts of %d (%f))\n",
                                      numUsers, numGoldUsers, numGoldPosts, numPosts,
                                      (double)numGoldPosts / numPosts);
                }

                // If we have any gold data for this user, pull the user out and
                // add them to one of the folds
                if (goldPosts.isEmpty())
                    continue;

                int whichFold = (int)(numGoldUsers % numFolds);
                numGoldUsers++;
                numGoldPosts += goldPosts.size();

                JSONObject userTestData = new JSONObject();
                userTestData.put("user_id", uid);
                JSONArray goldPostsArr = new JSONArray();
                for (JSONObject post : goldPosts) {
                    goldPostsArr.put(post);
                    long postId = post.getLong("id");
                    postIdPws[whichFold].println(postId);
                    JSONObject geo = post.getJSONObject("geo");
                    JSONArray coords = geo.getJSONArray("coordinates");
                    double lat = coords.getDouble(0);
                    double lon = coords.getDouble(1);
                    goldLocPws[whichFold].println(postId + "\t" + lat + "\t" + lon);
                }
                userTestData.put("posts", goldPostsArr);

                testDataPws[whichFold].println(userTestData);
                userIdPws[whichFold].println(uid);
                

            } catch (JSONException je) {
                je.printStackTrace();
            }
        }
        br.close();

        for (PrintWriter pw : postIdPws)
            pw.close();
        for (PrintWriter pw : userIdPws)
            pw.close();
        for (PrintWriter pw : goldLocPws)
            pw.close();
        for (PrintWriter pw : testDataPws)
            pw.close();
    }

    private static void loadInputFiles(File f, List<File> inputFiles) {
        if (f.isDirectory()) {
            for (File f2 : f.listFiles())
                loadInputFiles(f2, inputFiles);
        }
        else if (f.getName().endsWith(".gz"))
            inputFiles.add(f);
    }

    private static ArgOptions createArgOpts(String args[]) {
        ArgOptions opts = new ArgOptions();
        
        opts.addOption('h', "help", "Generates a help message and exits",
                          false, null, "Program Options");

        opts.addOption('f', "force", "Overwrite the existing datset",
                          false, null, "Program Options");

        opts.addOption('s', "sparse", "Creates a sparse version of the datset in addition to the dense dataset (default: true)",
                       true, "Boolean", "Output Options");

        opts.addOption('w', "weighted-network", "Creates a weighted social network (default: true)",
                       true, "Boolean", "Output Options");

        opts.addOption('t', "temporal-network", "Creates a temporally-annotated social network (default: false)",
                       true, "Boolean", "Output Options");

        opts.addOption('d', "directed-network", "Creates a directed social network (default: true)",
                       true, "Boolean", "Output Options");


        opts.addOption('c', "create-cross-validation", "Splits the data into folds for cross-validation testing",
                       true, "DIR", "Cross validation Options");

        opts.addOption('F', "cv-folds", "specifies the number of cross-validation folds (default: 5)",
                       true, "Int", "Cross validation Options");


        opts.addOption('N', "network-in-memory", "Creates the network in memory, which is faster but requires much more RAM (default: true)",
                       true, "Boolean", "Run-time Options");

        opts.addOption('P', "posts-in-memory", "Creates the post and user files in memory, which is faster but requires HUGE amounts of RAM (default: false)",
                       true, "Boolean", "Run-time Options");
        opts.addOption('T', "tmp-dir", "Uses the specified directory for all temporary files",
                       true, "File", "Run-time Options");

        opts.parseOptions(args);
        return opts;
    }


    static BufferedReader openGz(File f) throws IOException {
        return new BufferedReader(
            new InputStreamReader(new GZIPInputStream(
            new BufferedInputStream(new FileInputStream(f)))));
    }

    static PrintWriter openGzWrite(File f) throws IOException {
        return new PrintWriter(
            new BufferedOutputStream(new GZIPOutputStream(
            new FileOutputStream(f))));
    }

   static PrintWriter openGzAppend(File f) throws IOException {
        return new PrintWriter(
            new BufferedOutputStream(new GZIPOutputStream(
            new FileOutputStream(f, true))));
    }

}
