package edu.uw.bhi.bionlp.covid.parser;

import edu.uw.bhi.bionlp.covid.parser.grpcclient.AssertionClassifierChannelManager;
import io.grpc.*;

public class App 
{
    public static void main( String[] args ) throws Exception
    {
      // Generate single share-able assertion classifier channel.
      AssertionClassifierChannelManager assertionClassifierChannelManager = new AssertionClassifierChannelManager();

      // Build server
      Server server = ServerBuilder.forPort(Integer.parseInt(System.getenv("METAMAP_PORT")))
        .addService(new MetaMapImpl(assertionClassifierChannelManager))
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