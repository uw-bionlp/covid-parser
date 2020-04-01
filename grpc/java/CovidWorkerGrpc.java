package edu.uw.bhi.bionlp.covidworker;

import static io.grpc.stub.ClientCalls.asyncUnaryCall;
import static io.grpc.stub.ClientCalls.asyncServerStreamingCall;
import static io.grpc.stub.ClientCalls.asyncClientStreamingCall;
import static io.grpc.stub.ClientCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.blockingServerStreamingCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncServerStreamingCall;
import static io.grpc.stub.ServerCalls.asyncClientStreamingCall;
import static io.grpc.stub.ServerCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.7.0)",
    comments = "Source: CovidWorker.proto")
public final class CovidWorkerGrpc {

  private CovidWorkerGrpc() {}

  public static final String SERVICE_NAME = "covidworker.CovidWorker";

  // Static method descriptors that strictly reflect the proto.
  @io.grpc.ExperimentalApi("https://github.com/grpc/grpc-java/issues/1901")
  public static final io.grpc.MethodDescriptor<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload,
      edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> METHOD_RUN_META_MAP =
      io.grpc.MethodDescriptor.<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload, edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>newBuilder()
          .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
          .setFullMethodName(generateFullMethodName(
              "covidworker.CovidWorker", "RunMetaMap"))
          .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload.getDefaultInstance()))
          .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result.getDefaultInstance()))
          .setSchemaDescriptor(new CovidWorkerMethodDescriptorSupplier("RunMetaMap"))
          .build();
  @io.grpc.ExperimentalApi("https://github.com/grpc/grpc-java/issues/1901")
  public static final io.grpc.MethodDescriptor<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload,
      edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> METHOD_RUN_COVID_PREDICTOR =
      io.grpc.MethodDescriptor.<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload, edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>newBuilder()
          .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
          .setFullMethodName(generateFullMethodName(
              "covidworker.CovidWorker", "RunCovidPredictor"))
          .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload.getDefaultInstance()))
          .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
              edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result.getDefaultInstance()))
          .setSchemaDescriptor(new CovidWorkerMethodDescriptorSupplier("RunCovidPredictor"))
          .build();

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static CovidWorkerStub newStub(io.grpc.Channel channel) {
    return new CovidWorkerStub(channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static CovidWorkerBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    return new CovidWorkerBlockingStub(channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static CovidWorkerFutureStub newFutureStub(
      io.grpc.Channel channel) {
    return new CovidWorkerFutureStub(channel);
  }

  /**
   */
  public static abstract class CovidWorkerImplBase implements io.grpc.BindableService {

    /**
     */
    public void runMetaMap(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request,
        io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> responseObserver) {
      asyncUnimplementedUnaryCall(METHOD_RUN_META_MAP, responseObserver);
    }

    /**
     */
    public void runCovidPredictor(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request,
        io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> responseObserver) {
      asyncUnimplementedUnaryCall(METHOD_RUN_COVID_PREDICTOR, responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            METHOD_RUN_META_MAP,
            asyncUnaryCall(
              new MethodHandlers<
                edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload,
                edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>(
                  this, METHODID_RUN_META_MAP)))
          .addMethod(
            METHOD_RUN_COVID_PREDICTOR,
            asyncUnaryCall(
              new MethodHandlers<
                edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload,
                edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>(
                  this, METHODID_RUN_COVID_PREDICTOR)))
          .build();
    }
  }

  /**
   */
  public static final class CovidWorkerStub extends io.grpc.stub.AbstractStub<CovidWorkerStub> {
    private CovidWorkerStub(io.grpc.Channel channel) {
      super(channel);
    }

    private CovidWorkerStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CovidWorkerStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new CovidWorkerStub(channel, callOptions);
    }

    /**
     */
    public void runMetaMap(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request,
        io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(METHOD_RUN_META_MAP, getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void runCovidPredictor(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request,
        io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(METHOD_RUN_COVID_PREDICTOR, getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class CovidWorkerBlockingStub extends io.grpc.stub.AbstractStub<CovidWorkerBlockingStub> {
    private CovidWorkerBlockingStub(io.grpc.Channel channel) {
      super(channel);
    }

    private CovidWorkerBlockingStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CovidWorkerBlockingStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new CovidWorkerBlockingStub(channel, callOptions);
    }

    /**
     */
    public edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result runMetaMap(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request) {
      return blockingUnaryCall(
          getChannel(), METHOD_RUN_META_MAP, getCallOptions(), request);
    }

    /**
     */
    public edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result runCovidPredictor(edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request) {
      return blockingUnaryCall(
          getChannel(), METHOD_RUN_COVID_PREDICTOR, getCallOptions(), request);
    }
  }

  /**
   */
  public static final class CovidWorkerFutureStub extends io.grpc.stub.AbstractStub<CovidWorkerFutureStub> {
    private CovidWorkerFutureStub(io.grpc.Channel channel) {
      super(channel);
    }

    private CovidWorkerFutureStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CovidWorkerFutureStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new CovidWorkerFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> runMetaMap(
        edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request) {
      return futureUnaryCall(
          getChannel().newCall(METHOD_RUN_META_MAP, getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result> runCovidPredictor(
        edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload request) {
      return futureUnaryCall(
          getChannel().newCall(METHOD_RUN_COVID_PREDICTOR, getCallOptions()), request);
    }
  }

  private static final int METHODID_RUN_META_MAP = 0;
  private static final int METHODID_RUN_COVID_PREDICTOR = 1;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final CovidWorkerImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(CovidWorkerImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_RUN_META_MAP:
          serviceImpl.runMetaMap((edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload) request,
              (io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>) responseObserver);
          break;
        case METHODID_RUN_COVID_PREDICTOR:
          serviceImpl.runCovidPredictor((edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Payload) request,
              (io.grpc.stub.StreamObserver<edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.Result>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class CovidWorkerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    CovidWorkerBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("CovidWorker");
    }
  }

  private static final class CovidWorkerFileDescriptorSupplier
      extends CovidWorkerBaseDescriptorSupplier {
    CovidWorkerFileDescriptorSupplier() {}
  }

  private static final class CovidWorkerMethodDescriptorSupplier
      extends CovidWorkerBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    CovidWorkerMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (CovidWorkerGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new CovidWorkerFileDescriptorSupplier())
              .addMethod(METHOD_RUN_META_MAP)
              .addMethod(METHOD_RUN_COVID_PREDICTOR)
              .build();
        }
      }
    }
    return result;
  }
}
