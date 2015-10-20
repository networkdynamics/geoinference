# Dependencies

For this library to be useful, you will need the following:

1. `maven` installed: `sudo apt-get update && sudo apt-get install maven`
2. Java 8 installed:

    ```
    sudo apt-get install python-software-properties
    sudo add-apt-repository ppa:openjdk-r/ppa
    sudo apt-get update && sudo apt-get install openjdk-8-jdk
    sudo update-alternatives --config java # select openjdk-8
    sudo update-alternatives --config javac # select openjdk-8
    ```

3. A large dataset to work with. One configuration that is known to work is directories filled with `tweet-mm-dd-yy.txt.gz` files with individual tweets on seperate lines in JSON format.
4. A source of __ground truth__ data: A text file that has lines in the format of `<userid> <lat> <long>` e.g. `325390424 30.618972 -96.338798`

# Getting Started

1. Clone the repository with `git clone`
2. `cd <repo>/java`
3. Build the jars: `sudo mvn package`
4. Preprocess your data:

    ```
    java -cp target/geoinference-1.0.0-jar-with-dependencies.jar \
    ca.mcgill.networkdynamics.geoinference.CreateDataset \
    <processed-output-dir> <data dir>
    ```

5. Create a training model:

    ```
    java -jar target/geoinference-1.0.0-jar-with-dependencies.jar train \
    ca.mcgill.networkdynamics.geoinference.tv.TotalVariation  \
    <processed-data-dir> <output-dir>
    ```

6. Use the model to infer:

    ```
    java -jar target/geoinference-1.0.0-jar-with-dependencies.jar infer \
    ca.mcgill.networkdynamics.geoinference.tv.TotalVariation \
    <trained-model-dir> <data-dir> <inferences-output-file>
    ```
