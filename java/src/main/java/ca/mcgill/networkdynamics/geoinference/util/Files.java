/*
 * Copyright (c) 2015, David Jurgens
 *
 * All rights reserved. See LICENSE file for details
 */

package ca.mcgill.networkdynamics.geoinference.util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;


/**
 * A utility class for working with files.
 */
public class Files {

    public static BufferedReader openGz(File f) throws IOException {
        return new BufferedReader(
            new InputStreamReader(new GZIPInputStream(
            new BufferedInputStream(new FileInputStream(f)))));
    }

    public static PrintWriter openGzWrite(File f) throws IOException {
        return new PrintWriter(
            new BufferedOutputStream(new GZIPOutputStream(
            new FileOutputStream(f))));
    }


}
