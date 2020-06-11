package edu.uw.bhi.bionlp.covid.parser;

import io.grpc.*;

public class App 
{
    public static void main( String[] args ) throws Exception
    {
      Server server = ServerBuilder.forPort(Integer.parseInt(System.getenv("OPENNLP_PORT")))
        .addService(new OpenNLPImpl())
        .build();

      // Start the server
      server.start();

      // Server threads are running in the background.
      System.out.println("OpenNLP is up!");

      // Don't exit the main thread. Wait until server is terminated.
      server.awaitTermination();
    }
}

// Test with: 
// mvn -DskipTests package exec:java -Dexec.mainClass=edu.uw.bhi.bionlp.covid.parser.App