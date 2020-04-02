package edu.uw.bhi.bionlp.covid.parser;

import io.grpc.*;

public class App 
{
    public static void main( String[] args ) throws Exception
    {
      Server server = ServerBuilder.forPort(5000)
        .addService(new MetaMapImpl())
        .build();

      // Start the server
      server.start();

      // Server threads are running in the background.
      System.out.println("MetaMap is up!");

      // Don't exit the main thread. Wait until server is terminated.
      server.awaitTermination();
    }
}

// Test with: 
// mvn -DskipTests package exec:java -Dexec.mainClass=edu.uw.bhi.bionlp.covid.parser.App