/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference;

import java.io.File;

import org.json.JSONObject;


/**
 * The inferface for geoinference methods which specifies how they are to be
 * trained and how a trained model should be loaded from disk.
 */
public interface GeoinferenceMethod {

    /**
     * Trains a model using the posts contained in the dataset and returns the
     * trained model, optionally saving the model in the specified directory if
     * it is not {@code null}.
     *
     * @param settings a JSON object provided by the called that contains
     *        method-specific parameters needed for training (such as parameter
     *        values of the locations of other resources on disk.).
     * @param dataset the dataset to be used in training
     * @param modelDir the directory to which the trainined model should be
     *        saved or {@code null} if the trained model should be saved to
     *        disk.
     */
    GeoinferenceModel train(JSONObject settings, Dataset dataset, File modelDir);

    /**
     * Loads an already-trained model from its state on disk
     *
     * @param settings a JSON object provided by the called that contains
     *        method-specific parameters needed for training (such as parameter
     *        values of the locations of other resources on disk.).
     * @param modelDir the directory from which the trainined model should be
     *        loaded.
     */
    GeoinferenceModel load(JSONObject settings, File modelDir);
}
