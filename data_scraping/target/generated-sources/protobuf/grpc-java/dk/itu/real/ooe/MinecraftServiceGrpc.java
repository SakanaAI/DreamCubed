package dk.itu.real.ooe;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ClientCalls.asyncClientStreamingCall;
import static io.grpc.stub.ClientCalls.asyncServerStreamingCall;
import static io.grpc.stub.ClientCalls.asyncUnaryCall;
import static io.grpc.stub.ClientCalls.blockingServerStreamingCall;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ServerCalls.asyncClientStreamingCall;
import static io.grpc.stub.ServerCalls.asyncServerStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 * <pre>
 **
 *The main service.
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.21.0)",
    comments = "Source: minecraft.proto")
public final class MinecraftServiceGrpc {

  private MinecraftServiceGrpc() {}

  public static final String SERVICE_NAME = "dk.itu.real.ooe.MinecraftService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Blocks,
      com.google.protobuf.Empty> getSpawnBlocksMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "spawnBlocks",
      requestType = dk.itu.real.ooe.Minecraft.Blocks.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Blocks,
      com.google.protobuf.Empty> getSpawnBlocksMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Blocks, com.google.protobuf.Empty> getSpawnBlocksMethod;
    if ((getSpawnBlocksMethod = MinecraftServiceGrpc.getSpawnBlocksMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSpawnBlocksMethod = MinecraftServiceGrpc.getSpawnBlocksMethod) == null) {
          MinecraftServiceGrpc.getSpawnBlocksMethod = getSpawnBlocksMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Blocks, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "spawnBlocks"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Blocks.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("spawnBlocks"))
                  .build();
          }
        }
     }
     return getSpawnBlocksMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Uuids,
      dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readEntities",
      requestType = dk.itu.real.ooe.Minecraft.Uuids.class,
      responseType = dk.itu.real.ooe.Minecraft.Entities.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Uuids,
      dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Uuids, dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesMethod;
    if ((getReadEntitiesMethod = MinecraftServiceGrpc.getReadEntitiesMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadEntitiesMethod = MinecraftServiceGrpc.getReadEntitiesMethod) == null) {
          MinecraftServiceGrpc.getReadEntitiesMethod = getReadEntitiesMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Uuids, dk.itu.real.ooe.Minecraft.Entities>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readEntities"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Uuids.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Entities.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readEntities"))
                  .build();
          }
        }
     }
     return getReadEntitiesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SpawnEntities,
      dk.itu.real.ooe.Minecraft.Uuids> getSpawnEntitiesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "spawnEntities",
      requestType = dk.itu.real.ooe.Minecraft.SpawnEntities.class,
      responseType = dk.itu.real.ooe.Minecraft.Uuids.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SpawnEntities,
      dk.itu.real.ooe.Minecraft.Uuids> getSpawnEntitiesMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SpawnEntities, dk.itu.real.ooe.Minecraft.Uuids> getSpawnEntitiesMethod;
    if ((getSpawnEntitiesMethod = MinecraftServiceGrpc.getSpawnEntitiesMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSpawnEntitiesMethod = MinecraftServiceGrpc.getSpawnEntitiesMethod) == null) {
          MinecraftServiceGrpc.getSpawnEntitiesMethod = getSpawnEntitiesMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.SpawnEntities, dk.itu.real.ooe.Minecraft.Uuids>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "spawnEntities"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.SpawnEntities.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Uuids.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("spawnEntities"))
                  .build();
          }
        }
     }
     return getSpawnEntitiesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readCube",
      requestType = dk.itu.real.ooe.Minecraft.Cube.class,
      responseType = dk.itu.real.ooe.Minecraft.Blocks.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks> getReadCubeMethod;
    if ((getReadCubeMethod = MinecraftServiceGrpc.getReadCubeMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadCubeMethod = MinecraftServiceGrpc.getReadCubeMethod) == null) {
          MinecraftServiceGrpc.getReadCubeMethod = getReadCubeMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readCube"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Cube.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Blocks.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readCube"))
                  .build();
          }
        }
     }
     return getReadCubeMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.FillCubeRequest,
      com.google.protobuf.Empty> getFillCubeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "fillCube",
      requestType = dk.itu.real.ooe.Minecraft.FillCubeRequest.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.FillCubeRequest,
      com.google.protobuf.Empty> getFillCubeMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.FillCubeRequest, com.google.protobuf.Empty> getFillCubeMethod;
    if ((getFillCubeMethod = MinecraftServiceGrpc.getFillCubeMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getFillCubeMethod = MinecraftServiceGrpc.getFillCubeMethod) == null) {
          MinecraftServiceGrpc.getFillCubeMethod = getFillCubeMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.FillCubeRequest, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "fillCube"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.FillCubeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("fillCube"))
                  .build();
          }
        }
     }
     return getFillCubeMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Sphere,
      dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesInSphereMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readEntitiesInSphere",
      requestType = dk.itu.real.ooe.Minecraft.Sphere.class,
      responseType = dk.itu.real.ooe.Minecraft.Entities.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Sphere,
      dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesInSphereMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Sphere, dk.itu.real.ooe.Minecraft.Entities> getReadEntitiesInSphereMethod;
    if ((getReadEntitiesInSphereMethod = MinecraftServiceGrpc.getReadEntitiesInSphereMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadEntitiesInSphereMethod = MinecraftServiceGrpc.getReadEntitiesInSphereMethod) == null) {
          MinecraftServiceGrpc.getReadEntitiesInSphereMethod = getReadEntitiesInSphereMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Sphere, dk.itu.real.ooe.Minecraft.Entities>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readEntitiesInSphere"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Sphere.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Entities.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readEntitiesInSphere"))
                  .build();
          }
        }
     }
     return getReadEntitiesInSphereMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.BiomeResponse> getGetBiomeAtMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getBiomeAt",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = dk.itu.real.ooe.Minecraft.BiomeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.BiomeResponse> getGetBiomeAtMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.BiomeResponse> getGetBiomeAtMethod;
    if ((getGetBiomeAtMethod = MinecraftServiceGrpc.getGetBiomeAtMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getGetBiomeAtMethod = MinecraftServiceGrpc.getGetBiomeAtMethod) == null) {
          MinecraftServiceGrpc.getGetBiomeAtMethod = getGetBiomeAtMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.BiomeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "getBiomeAt"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.BiomeResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("getBiomeAt"))
                  .build();
          }
        }
     }
     return getGetBiomeAtMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getSetLocMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setLoc",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = dk.itu.real.ooe.Minecraft.Point.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getSetLocMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point> getSetLocMethod;
    if ((getSetLocMethod = MinecraftServiceGrpc.getSetLocMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetLocMethod = MinecraftServiceGrpc.getSetLocMethod) == null) {
          MinecraftServiceGrpc.getSetLocMethod = getSetLocMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setLoc"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setLoc"))
                  .build();
          }
        }
     }
     return getSetLocMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      com.google.protobuf.Empty> getSetRotMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setRot",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      com.google.protobuf.Empty> getSetRotMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, com.google.protobuf.Empty> getSetRotMethod;
    if ((getSetRotMethod = MinecraftServiceGrpc.getSetRotMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetRotMethod = MinecraftServiceGrpc.getSetRotMethod) == null) {
          MinecraftServiceGrpc.getSetRotMethod = getSetRotMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setRot"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setRot"))
                  .build();
          }
        }
     }
     return getSetRotMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LocRotMsg,
      com.google.protobuf.Empty> getSetPlayerLocRotMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setPlayerLocRot",
      requestType = dk.itu.real.ooe.Minecraft.LocRotMsg.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LocRotMsg,
      com.google.protobuf.Empty> getSetPlayerLocRotMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LocRotMsg, com.google.protobuf.Empty> getSetPlayerLocRotMethod;
    if ((getSetPlayerLocRotMethod = MinecraftServiceGrpc.getSetPlayerLocRotMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetPlayerLocRotMethod = MinecraftServiceGrpc.getSetPlayerLocRotMethod) == null) {
          MinecraftServiceGrpc.getSetPlayerLocRotMethod = getSetPlayerLocRotMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.LocRotMsg, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setPlayerLocRot"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.LocRotMsg.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setPlayerLocRot"))
                  .build();
          }
        }
     }
     return getSetPlayerLocRotMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      com.google.protobuf.Empty> getInitDataGenMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "initDataGen",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      com.google.protobuf.Empty> getInitDataGenMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, com.google.protobuf.Empty> getInitDataGenMethod;
    if ((getInitDataGenMethod = MinecraftServiceGrpc.getInitDataGenMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getInitDataGenMethod = MinecraftServiceGrpc.getInitDataGenMethod) == null) {
          MinecraftServiceGrpc.getInitDataGenMethod = getInitDataGenMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "initDataGen"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("initDataGen"))
                  .build();
          }
        }
     }
     return getInitDataGenMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getGetHighestYAtMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getHighestYAt",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = dk.itu.real.ooe.Minecraft.Point.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getGetHighestYAtMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point> getGetHighestYAtMethod;
    if ((getGetHighestYAtMethod = MinecraftServiceGrpc.getGetHighestYAtMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getGetHighestYAtMethod = MinecraftServiceGrpc.getGetHighestYAtMethod) == null) {
          MinecraftServiceGrpc.getGetHighestYAtMethod = getGetHighestYAtMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "getHighestYAt"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("getHighestYAt"))
                  .build();
          }
        }
     }
     return getGetHighestYAtMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getSetLocYMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setLocY",
      requestType = dk.itu.real.ooe.Minecraft.Point.class,
      responseType = dk.itu.real.ooe.Minecraft.Point.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point,
      dk.itu.real.ooe.Minecraft.Point> getSetLocYMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point> getSetLocYMethod;
    if ((getSetLocYMethod = MinecraftServiceGrpc.getSetLocYMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetLocYMethod = MinecraftServiceGrpc.getSetLocYMethod) == null) {
          MinecraftServiceGrpc.getSetLocYMethod = getSetLocYMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Point, dk.itu.real.ooe.Minecraft.Point>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setLocY"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Point.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setLocY"))
                  .build();
          }
        }
     }
     return getSetLocYMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readCubeAndBiome",
      requestType = dk.itu.real.ooe.Minecraft.Cube.class,
      responseType = dk.itu.real.ooe.Minecraft.Blocks.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMethod;
    if ((getReadCubeAndBiomeMethod = MinecraftServiceGrpc.getReadCubeAndBiomeMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadCubeAndBiomeMethod = MinecraftServiceGrpc.getReadCubeAndBiomeMethod) == null) {
          MinecraftServiceGrpc.getReadCubeAndBiomeMethod = getReadCubeAndBiomeMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readCubeAndBiome"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Cube.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Blocks.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readCubeAndBiome"))
                  .build();
          }
        }
     }
     return getReadCubeAndBiomeMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMetadataMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readCubeAndBiomeMetadata",
      requestType = dk.itu.real.ooe.Minecraft.Cube.class,
      responseType = dk.itu.real.ooe.Minecraft.Blocks.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMetadataMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks> getReadCubeAndBiomeMetadataMethod;
    if ((getReadCubeAndBiomeMetadataMethod = MinecraftServiceGrpc.getReadCubeAndBiomeMetadataMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadCubeAndBiomeMetadataMethod = MinecraftServiceGrpc.getReadCubeAndBiomeMetadataMethod) == null) {
          MinecraftServiceGrpc.getReadCubeAndBiomeMetadataMethod = getReadCubeAndBiomeMetadataMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.Blocks>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readCubeAndBiomeMetadata"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Cube.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Blocks.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readCubeAndBiomeMetadata"))
                  .build();
          }
        }
     }
     return getReadCubeAndBiomeMetadataMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.DenseCube> getReadDenseCubeWithMetadataMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readDenseCubeWithMetadata",
      requestType = dk.itu.real.ooe.Minecraft.Cube.class,
      responseType = dk.itu.real.ooe.Minecraft.DenseCube.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.DenseCube> getReadDenseCubeWithMetadataMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.DenseCube> getReadDenseCubeWithMetadataMethod;
    if ((getReadDenseCubeWithMetadataMethod = MinecraftServiceGrpc.getReadDenseCubeWithMetadataMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadDenseCubeWithMetadataMethod = MinecraftServiceGrpc.getReadDenseCubeWithMetadataMethod) == null) {
          MinecraftServiceGrpc.getReadDenseCubeWithMetadataMethod = getReadDenseCubeWithMetadataMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.DenseCube>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readDenseCubeWithMetadata"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Cube.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.DenseCube.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readDenseCubeWithMetadata"))
                  .build();
          }
        }
     }
     return getReadDenseCubeWithMetadataMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.DenseCubeMajority> getReadDenseCubeWithMajorityMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "readDenseCubeWithMajority",
      requestType = dk.itu.real.ooe.Minecraft.Cube.class,
      responseType = dk.itu.real.ooe.Minecraft.DenseCubeMajority.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube,
      dk.itu.real.ooe.Minecraft.DenseCubeMajority> getReadDenseCubeWithMajorityMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.DenseCubeMajority> getReadDenseCubeWithMajorityMethod;
    if ((getReadDenseCubeWithMajorityMethod = MinecraftServiceGrpc.getReadDenseCubeWithMajorityMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getReadDenseCubeWithMajorityMethod = MinecraftServiceGrpc.getReadDenseCubeWithMajorityMethod) == null) {
          MinecraftServiceGrpc.getReadDenseCubeWithMajorityMethod = getReadDenseCubeWithMajorityMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.Cube, dk.itu.real.ooe.Minecraft.DenseCubeMajority>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "readDenseCubeWithMajority"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.Cube.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.DenseCubeMajority.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("readDenseCubeWithMajority"))
                  .build();
          }
        }
     }
     return getReadDenseCubeWithMajorityMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SetAreaRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getSetModelAreaMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setModelArea",
      requestType = dk.itu.real.ooe.Minecraft.SetAreaRequest.class,
      responseType = dk.itu.real.ooe.Minecraft.TriggerResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SetAreaRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getSetModelAreaMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.SetAreaRequest, dk.itu.real.ooe.Minecraft.TriggerResponse> getSetModelAreaMethod;
    if ((getSetModelAreaMethod = MinecraftServiceGrpc.getSetModelAreaMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetModelAreaMethod = MinecraftServiceGrpc.getSetModelAreaMethod) == null) {
          MinecraftServiceGrpc.getSetModelAreaMethod = getSetModelAreaMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.SetAreaRequest, dk.itu.real.ooe.Minecraft.TriggerResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setModelArea"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.SetAreaRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TriggerResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setModelArea"))
                  .build();
          }
        }
     }
     return getSetModelAreaMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      dk.itu.real.ooe.Minecraft.BuildingArea> getGetCurrentBuildingAreaMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getCurrentBuildingArea",
      requestType = com.google.protobuf.Empty.class,
      responseType = dk.itu.real.ooe.Minecraft.BuildingArea.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      dk.itu.real.ooe.Minecraft.BuildingArea> getGetCurrentBuildingAreaMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, dk.itu.real.ooe.Minecraft.BuildingArea> getGetCurrentBuildingAreaMethod;
    if ((getGetCurrentBuildingAreaMethod = MinecraftServiceGrpc.getGetCurrentBuildingAreaMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getGetCurrentBuildingAreaMethod = MinecraftServiceGrpc.getGetCurrentBuildingAreaMethod) == null) {
          MinecraftServiceGrpc.getGetCurrentBuildingAreaMethod = getGetCurrentBuildingAreaMethod = 
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, dk.itu.real.ooe.Minecraft.BuildingArea>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "getCurrentBuildingArea"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.BuildingArea.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("getCurrentBuildingArea"))
                  .build();
          }
        }
     }
     return getGetCurrentBuildingAreaMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getTriggerModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "triggerModel",
      requestType = com.google.protobuf.Empty.class,
      responseType = dk.itu.real.ooe.Minecraft.TriggerResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getTriggerModelMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, dk.itu.real.ooe.Minecraft.TriggerResponse> getTriggerModelMethod;
    if ((getTriggerModelMethod = MinecraftServiceGrpc.getTriggerModelMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getTriggerModelMethod = MinecraftServiceGrpc.getTriggerModelMethod) == null) {
          MinecraftServiceGrpc.getTriggerModelMethod = getTriggerModelMethod = 
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, dk.itu.real.ooe.Minecraft.TriggerResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "triggerModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TriggerResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("triggerModel"))
                  .build();
          }
        }
     }
     return getTriggerModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LoadPresetRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getHandleLoadPresetMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "HandleLoadPreset",
      requestType = dk.itu.real.ooe.Minecraft.LoadPresetRequest.class,
      responseType = dk.itu.real.ooe.Minecraft.TriggerResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LoadPresetRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getHandleLoadPresetMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.LoadPresetRequest, dk.itu.real.ooe.Minecraft.TriggerResponse> getHandleLoadPresetMethod;
    if ((getHandleLoadPresetMethod = MinecraftServiceGrpc.getHandleLoadPresetMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getHandleLoadPresetMethod = MinecraftServiceGrpc.getHandleLoadPresetMethod) == null) {
          MinecraftServiceGrpc.getHandleLoadPresetMethod = getHandleLoadPresetMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.LoadPresetRequest, dk.itu.real.ooe.Minecraft.TriggerResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "HandleLoadPreset"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.LoadPresetRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TriggerResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("HandleLoadPreset"))
                  .build();
          }
        }
     }
     return getHandleLoadPresetMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerrainModificationRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getModifyTerrainMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModifyTerrain",
      requestType = dk.itu.real.ooe.Minecraft.TerrainModificationRequest.class,
      responseType = dk.itu.real.ooe.Minecraft.TriggerResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerrainModificationRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getModifyTerrainMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerrainModificationRequest, dk.itu.real.ooe.Minecraft.TriggerResponse> getModifyTerrainMethod;
    if ((getModifyTerrainMethod = MinecraftServiceGrpc.getModifyTerrainMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getModifyTerrainMethod = MinecraftServiceGrpc.getModifyTerrainMethod) == null) {
          MinecraftServiceGrpc.getModifyTerrainMethod = getModifyTerrainMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.TerrainModificationRequest, dk.itu.real.ooe.Minecraft.TriggerResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "ModifyTerrain"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TerrainModificationRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TriggerResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("ModifyTerrain"))
                  .build();
          }
        }
     }
     return getModifyTerrainMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerraformRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getTerraformMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Terraform",
      requestType = dk.itu.real.ooe.Minecraft.TerraformRequest.class,
      responseType = dk.itu.real.ooe.Minecraft.TriggerResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerraformRequest,
      dk.itu.real.ooe.Minecraft.TriggerResponse> getTerraformMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.TerraformRequest, dk.itu.real.ooe.Minecraft.TriggerResponse> getTerraformMethod;
    if ((getTerraformMethod = MinecraftServiceGrpc.getTerraformMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getTerraformMethod = MinecraftServiceGrpc.getTerraformMethod) == null) {
          MinecraftServiceGrpc.getTerraformMethod = getTerraformMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.TerraformRequest, dk.itu.real.ooe.Minecraft.TriggerResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "Terraform"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TerraformRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.TriggerResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("Terraform"))
                  .build();
          }
        }
     }
     return getTerraformMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.DimensionRequest,
      com.google.protobuf.Empty> getSetActiveDimensionMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "setActiveDimension",
      requestType = dk.itu.real.ooe.Minecraft.DimensionRequest.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.DimensionRequest,
      com.google.protobuf.Empty> getSetActiveDimensionMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.DimensionRequest, com.google.protobuf.Empty> getSetActiveDimensionMethod;
    if ((getSetActiveDimensionMethod = MinecraftServiceGrpc.getSetActiveDimensionMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getSetActiveDimensionMethod = MinecraftServiceGrpc.getSetActiveDimensionMethod) == null) {
          MinecraftServiceGrpc.getSetActiveDimensionMethod = getSetActiveDimensionMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.DimensionRequest, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "setActiveDimension"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.DimensionRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("setActiveDimension"))
                  .build();
          }
        }
     }
     return getSetActiveDimensionMethod;
  }

  private static volatile io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.PreloadChunksRequest,
      com.google.protobuf.Empty> getPreloadChunksMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "preloadChunks",
      requestType = dk.itu.real.ooe.Minecraft.PreloadChunksRequest.class,
      responseType = com.google.protobuf.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.PreloadChunksRequest,
      com.google.protobuf.Empty> getPreloadChunksMethod() {
    io.grpc.MethodDescriptor<dk.itu.real.ooe.Minecraft.PreloadChunksRequest, com.google.protobuf.Empty> getPreloadChunksMethod;
    if ((getPreloadChunksMethod = MinecraftServiceGrpc.getPreloadChunksMethod) == null) {
      synchronized (MinecraftServiceGrpc.class) {
        if ((getPreloadChunksMethod = MinecraftServiceGrpc.getPreloadChunksMethod) == null) {
          MinecraftServiceGrpc.getPreloadChunksMethod = getPreloadChunksMethod = 
              io.grpc.MethodDescriptor.<dk.itu.real.ooe.Minecraft.PreloadChunksRequest, com.google.protobuf.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "dk.itu.real.ooe.MinecraftService", "preloadChunks"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  dk.itu.real.ooe.Minecraft.PreloadChunksRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
                  .setSchemaDescriptor(new MinecraftServiceMethodDescriptorSupplier("preloadChunks"))
                  .build();
          }
        }
     }
     return getPreloadChunksMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static MinecraftServiceStub newStub(io.grpc.Channel channel) {
    return new MinecraftServiceStub(channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static MinecraftServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    return new MinecraftServiceBlockingStub(channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static MinecraftServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    return new MinecraftServiceFutureStub(channel);
  }

  /**
   * <pre>
   **
   *The main service.
   * </pre>
   */
  public static abstract class MinecraftServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     ** Spawn multiple blocks. 
     * </pre>
     */
    public void spawnBlocks(dk.itu.real.ooe.Minecraft.Blocks request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getSpawnBlocksMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Reads multiple entities *
     * </pre>
     */
    public void readEntities(dk.itu.real.ooe.Minecraft.Uuids request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities> responseObserver) {
      asyncUnimplementedUnaryCall(getReadEntitiesMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Spawn multiple entities. 
     * </pre>
     */
    public void spawnEntities(dk.itu.real.ooe.Minecraft.SpawnEntities request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Uuids> responseObserver) {
      asyncUnimplementedUnaryCall(getSpawnEntitiesMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Return all blocks in a cube 
     * </pre>
     */
    public void readCube(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnimplementedUnaryCall(getReadCubeMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Fill a cube with a block type 
     * </pre>
     */
    public void fillCube(dk.itu.real.ooe.Minecraft.FillCubeRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getFillCubeMethod(), responseObserver);
    }

    /**
     */
    public void readEntitiesInSphere(dk.itu.real.ooe.Minecraft.Sphere request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities> responseObserver) {
      asyncUnimplementedUnaryCall(getReadEntitiesInSphereMethod(), responseObserver);
    }

    /**
     */
    public void getBiomeAt(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BiomeResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getGetBiomeAtMethod(), responseObserver);
    }

    /**
     */
    public void setLoc(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnimplementedUnaryCall(getSetLocMethod(), responseObserver);
    }

    /**
     */
    public void setRot(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getSetRotMethod(), responseObserver);
    }

    /**
     */
    public void setPlayerLocRot(dk.itu.real.ooe.Minecraft.LocRotMsg request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getSetPlayerLocRotMethod(), responseObserver);
    }

    /**
     */
    public void initDataGen(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getInitDataGenMethod(), responseObserver);
    }

    /**
     */
    public void getHighestYAt(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnimplementedUnaryCall(getGetHighestYAtMethod(), responseObserver);
    }

    /**
     */
    public void setLocY(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnimplementedUnaryCall(getSetLocYMethod(), responseObserver);
    }

    /**
     */
    public void readCubeAndBiome(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnimplementedUnaryCall(getReadCubeAndBiomeMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Return all blocks in a cube, including per-block metadata 
     * </pre>
     */
    public void readCubeAndBiomeMetadata(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnimplementedUnaryCall(getReadCubeAndBiomeMetadataMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + biome ids, plus sparse metadata for relevant blocks 
     * </pre>
     */
    public void readDenseCubeWithMetadata(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCube> responseObserver) {
      asyncUnimplementedUnaryCall(getReadDenseCubeWithMetadataMethod(), responseObserver);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + a single majority-biome label for the whole cube 
     * </pre>
     */
    public void readDenseCubeWithMajority(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCubeMajority> responseObserver) {
      asyncUnimplementedUnaryCall(getReadDenseCubeWithMajorityMethod(), responseObserver);
    }

    /**
     */
    public void setModelArea(dk.itu.real.ooe.Minecraft.SetAreaRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getSetModelAreaMethod(), responseObserver);
    }

    /**
     */
    public void getCurrentBuildingArea(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BuildingArea> responseObserver) {
      asyncUnimplementedUnaryCall(getGetCurrentBuildingAreaMethod(), responseObserver);
    }

    /**
     */
    public void triggerModel(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getTriggerModelMethod(), responseObserver);
    }

    /**
     */
    public void handleLoadPreset(dk.itu.real.ooe.Minecraft.LoadPresetRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getHandleLoadPresetMethod(), responseObserver);
    }

    /**
     */
    public void modifyTerrain(dk.itu.real.ooe.Minecraft.TerrainModificationRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getModifyTerrainMethod(), responseObserver);
    }

    /**
     * <pre>
     * Trigger diffusion-based terrain generation and spawning
     * </pre>
     */
    public void terraform(dk.itu.real.ooe.Minecraft.TerraformRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getTerraformMethod(), responseObserver);
    }

    /**
     * <pre>
     * Switch active dimension/world for subsequent RPC operations; optionally teleport player
     * </pre>
     */
    public void setActiveDimension(dk.itu.real.ooe.Minecraft.DimensionRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getSetActiveDimensionMethod(), responseObserver);
    }

    /**
     * <pre>
     * Force-load/generate chunks in the active dimension within a block-area
     * </pre>
     */
    public void preloadChunks(dk.itu.real.ooe.Minecraft.PreloadChunksRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnimplementedUnaryCall(getPreloadChunksMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getSpawnBlocksMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Blocks,
                com.google.protobuf.Empty>(
                  this, METHODID_SPAWN_BLOCKS)))
          .addMethod(
            getReadEntitiesMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Uuids,
                dk.itu.real.ooe.Minecraft.Entities>(
                  this, METHODID_READ_ENTITIES)))
          .addMethod(
            getSpawnEntitiesMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.SpawnEntities,
                dk.itu.real.ooe.Minecraft.Uuids>(
                  this, METHODID_SPAWN_ENTITIES)))
          .addMethod(
            getReadCubeMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Cube,
                dk.itu.real.ooe.Minecraft.Blocks>(
                  this, METHODID_READ_CUBE)))
          .addMethod(
            getFillCubeMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.FillCubeRequest,
                com.google.protobuf.Empty>(
                  this, METHODID_FILL_CUBE)))
          .addMethod(
            getReadEntitiesInSphereMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Sphere,
                dk.itu.real.ooe.Minecraft.Entities>(
                  this, METHODID_READ_ENTITIES_IN_SPHERE)))
          .addMethod(
            getGetBiomeAtMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                dk.itu.real.ooe.Minecraft.BiomeResponse>(
                  this, METHODID_GET_BIOME_AT)))
          .addMethod(
            getSetLocMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                dk.itu.real.ooe.Minecraft.Point>(
                  this, METHODID_SET_LOC)))
          .addMethod(
            getSetRotMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                com.google.protobuf.Empty>(
                  this, METHODID_SET_ROT)))
          .addMethod(
            getSetPlayerLocRotMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.LocRotMsg,
                com.google.protobuf.Empty>(
                  this, METHODID_SET_PLAYER_LOC_ROT)))
          .addMethod(
            getInitDataGenMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                com.google.protobuf.Empty>(
                  this, METHODID_INIT_DATA_GEN)))
          .addMethod(
            getGetHighestYAtMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                dk.itu.real.ooe.Minecraft.Point>(
                  this, METHODID_GET_HIGHEST_YAT)))
          .addMethod(
            getSetLocYMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Point,
                dk.itu.real.ooe.Minecraft.Point>(
                  this, METHODID_SET_LOC_Y)))
          .addMethod(
            getReadCubeAndBiomeMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Cube,
                dk.itu.real.ooe.Minecraft.Blocks>(
                  this, METHODID_READ_CUBE_AND_BIOME)))
          .addMethod(
            getReadCubeAndBiomeMetadataMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Cube,
                dk.itu.real.ooe.Minecraft.Blocks>(
                  this, METHODID_READ_CUBE_AND_BIOME_METADATA)))
          .addMethod(
            getReadDenseCubeWithMetadataMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Cube,
                dk.itu.real.ooe.Minecraft.DenseCube>(
                  this, METHODID_READ_DENSE_CUBE_WITH_METADATA)))
          .addMethod(
            getReadDenseCubeWithMajorityMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.Cube,
                dk.itu.real.ooe.Minecraft.DenseCubeMajority>(
                  this, METHODID_READ_DENSE_CUBE_WITH_MAJORITY)))
          .addMethod(
            getSetModelAreaMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.SetAreaRequest,
                dk.itu.real.ooe.Minecraft.TriggerResponse>(
                  this, METHODID_SET_MODEL_AREA)))
          .addMethod(
            getGetCurrentBuildingAreaMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                dk.itu.real.ooe.Minecraft.BuildingArea>(
                  this, METHODID_GET_CURRENT_BUILDING_AREA)))
          .addMethod(
            getTriggerModelMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                dk.itu.real.ooe.Minecraft.TriggerResponse>(
                  this, METHODID_TRIGGER_MODEL)))
          .addMethod(
            getHandleLoadPresetMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.LoadPresetRequest,
                dk.itu.real.ooe.Minecraft.TriggerResponse>(
                  this, METHODID_HANDLE_LOAD_PRESET)))
          .addMethod(
            getModifyTerrainMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.TerrainModificationRequest,
                dk.itu.real.ooe.Minecraft.TriggerResponse>(
                  this, METHODID_MODIFY_TERRAIN)))
          .addMethod(
            getTerraformMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.TerraformRequest,
                dk.itu.real.ooe.Minecraft.TriggerResponse>(
                  this, METHODID_TERRAFORM)))
          .addMethod(
            getSetActiveDimensionMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.DimensionRequest,
                com.google.protobuf.Empty>(
                  this, METHODID_SET_ACTIVE_DIMENSION)))
          .addMethod(
            getPreloadChunksMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                dk.itu.real.ooe.Minecraft.PreloadChunksRequest,
                com.google.protobuf.Empty>(
                  this, METHODID_PRELOAD_CHUNKS)))
          .build();
    }
  }

  /**
   * <pre>
   **
   *The main service.
   * </pre>
   */
  public static final class MinecraftServiceStub extends io.grpc.stub.AbstractStub<MinecraftServiceStub> {
    private MinecraftServiceStub(io.grpc.Channel channel) {
      super(channel);
    }

    private MinecraftServiceStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected MinecraftServiceStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new MinecraftServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     ** Spawn multiple blocks. 
     * </pre>
     */
    public void spawnBlocks(dk.itu.real.ooe.Minecraft.Blocks request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSpawnBlocksMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Reads multiple entities *
     * </pre>
     */
    public void readEntities(dk.itu.real.ooe.Minecraft.Uuids request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadEntitiesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Spawn multiple entities. 
     * </pre>
     */
    public void spawnEntities(dk.itu.real.ooe.Minecraft.SpawnEntities request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Uuids> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSpawnEntitiesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Return all blocks in a cube 
     * </pre>
     */
    public void readCube(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadCubeMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Fill a cube with a block type 
     * </pre>
     */
    public void fillCube(dk.itu.real.ooe.Minecraft.FillCubeRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getFillCubeMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void readEntitiesInSphere(dk.itu.real.ooe.Minecraft.Sphere request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadEntitiesInSphereMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getBiomeAt(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BiomeResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getGetBiomeAtMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void setLoc(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetLocMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void setRot(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetRotMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void setPlayerLocRot(dk.itu.real.ooe.Minecraft.LocRotMsg request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetPlayerLocRotMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void initDataGen(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getInitDataGenMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getHighestYAt(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getGetHighestYAtMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void setLocY(dk.itu.real.ooe.Minecraft.Point request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetLocYMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void readCubeAndBiome(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadCubeAndBiomeMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Return all blocks in a cube, including per-block metadata 
     * </pre>
     */
    public void readCubeAndBiomeMetadata(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadCubeAndBiomeMetadataMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + biome ids, plus sparse metadata for relevant blocks 
     * </pre>
     */
    public void readDenseCubeWithMetadata(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCube> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadDenseCubeWithMetadataMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + a single majority-biome label for the whole cube 
     * </pre>
     */
    public void readDenseCubeWithMajority(dk.itu.real.ooe.Minecraft.Cube request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCubeMajority> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getReadDenseCubeWithMajorityMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void setModelArea(dk.itu.real.ooe.Minecraft.SetAreaRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetModelAreaMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getCurrentBuildingArea(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BuildingArea> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getGetCurrentBuildingAreaMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void triggerModel(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getTriggerModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void handleLoadPreset(dk.itu.real.ooe.Minecraft.LoadPresetRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getHandleLoadPresetMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void modifyTerrain(dk.itu.real.ooe.Minecraft.TerrainModificationRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getModifyTerrainMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Trigger diffusion-based terrain generation and spawning
     * </pre>
     */
    public void terraform(dk.itu.real.ooe.Minecraft.TerraformRequest request,
        io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getTerraformMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Switch active dimension/world for subsequent RPC operations; optionally teleport player
     * </pre>
     */
    public void setActiveDimension(dk.itu.real.ooe.Minecraft.DimensionRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetActiveDimensionMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Force-load/generate chunks in the active dimension within a block-area
     * </pre>
     */
    public void preloadChunks(dk.itu.real.ooe.Minecraft.PreloadChunksRequest request,
        io.grpc.stub.StreamObserver<com.google.protobuf.Empty> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getPreloadChunksMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   **
   *The main service.
   * </pre>
   */
  public static final class MinecraftServiceBlockingStub extends io.grpc.stub.AbstractStub<MinecraftServiceBlockingStub> {
    private MinecraftServiceBlockingStub(io.grpc.Channel channel) {
      super(channel);
    }

    private MinecraftServiceBlockingStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected MinecraftServiceBlockingStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new MinecraftServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     ** Spawn multiple blocks. 
     * </pre>
     */
    public com.google.protobuf.Empty spawnBlocks(dk.itu.real.ooe.Minecraft.Blocks request) {
      return blockingUnaryCall(
          getChannel(), getSpawnBlocksMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Reads multiple entities *
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.Entities readEntities(dk.itu.real.ooe.Minecraft.Uuids request) {
      return blockingUnaryCall(
          getChannel(), getReadEntitiesMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Spawn multiple entities. 
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.Uuids spawnEntities(dk.itu.real.ooe.Minecraft.SpawnEntities request) {
      return blockingUnaryCall(
          getChannel(), getSpawnEntitiesMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Return all blocks in a cube 
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.Blocks readCube(dk.itu.real.ooe.Minecraft.Cube request) {
      return blockingUnaryCall(
          getChannel(), getReadCubeMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Fill a cube with a block type 
     * </pre>
     */
    public com.google.protobuf.Empty fillCube(dk.itu.real.ooe.Minecraft.FillCubeRequest request) {
      return blockingUnaryCall(
          getChannel(), getFillCubeMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.Entities readEntitiesInSphere(dk.itu.real.ooe.Minecraft.Sphere request) {
      return blockingUnaryCall(
          getChannel(), getReadEntitiesInSphereMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.BiomeResponse getBiomeAt(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getGetBiomeAtMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.Point setLoc(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getSetLocMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty setRot(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getSetRotMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty setPlayerLocRot(dk.itu.real.ooe.Minecraft.LocRotMsg request) {
      return blockingUnaryCall(
          getChannel(), getSetPlayerLocRotMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.google.protobuf.Empty initDataGen(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getInitDataGenMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.Point getHighestYAt(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getGetHighestYAtMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.Point setLocY(dk.itu.real.ooe.Minecraft.Point request) {
      return blockingUnaryCall(
          getChannel(), getSetLocYMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.Blocks readCubeAndBiome(dk.itu.real.ooe.Minecraft.Cube request) {
      return blockingUnaryCall(
          getChannel(), getReadCubeAndBiomeMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Return all blocks in a cube, including per-block metadata 
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.Blocks readCubeAndBiomeMetadata(dk.itu.real.ooe.Minecraft.Cube request) {
      return blockingUnaryCall(
          getChannel(), getReadCubeAndBiomeMetadataMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + biome ids, plus sparse metadata for relevant blocks 
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.DenseCube readDenseCubeWithMetadata(dk.itu.real.ooe.Minecraft.Cube request) {
      return blockingUnaryCall(
          getChannel(), getReadDenseCubeWithMetadataMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + a single majority-biome label for the whole cube 
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.DenseCubeMajority readDenseCubeWithMajority(dk.itu.real.ooe.Minecraft.Cube request) {
      return blockingUnaryCall(
          getChannel(), getReadDenseCubeWithMajorityMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.TriggerResponse setModelArea(dk.itu.real.ooe.Minecraft.SetAreaRequest request) {
      return blockingUnaryCall(
          getChannel(), getSetModelAreaMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.BuildingArea getCurrentBuildingArea(com.google.protobuf.Empty request) {
      return blockingUnaryCall(
          getChannel(), getGetCurrentBuildingAreaMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.TriggerResponse triggerModel(com.google.protobuf.Empty request) {
      return blockingUnaryCall(
          getChannel(), getTriggerModelMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.TriggerResponse handleLoadPreset(dk.itu.real.ooe.Minecraft.LoadPresetRequest request) {
      return blockingUnaryCall(
          getChannel(), getHandleLoadPresetMethod(), getCallOptions(), request);
    }

    /**
     */
    public dk.itu.real.ooe.Minecraft.TriggerResponse modifyTerrain(dk.itu.real.ooe.Minecraft.TerrainModificationRequest request) {
      return blockingUnaryCall(
          getChannel(), getModifyTerrainMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Trigger diffusion-based terrain generation and spawning
     * </pre>
     */
    public dk.itu.real.ooe.Minecraft.TriggerResponse terraform(dk.itu.real.ooe.Minecraft.TerraformRequest request) {
      return blockingUnaryCall(
          getChannel(), getTerraformMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Switch active dimension/world for subsequent RPC operations; optionally teleport player
     * </pre>
     */
    public com.google.protobuf.Empty setActiveDimension(dk.itu.real.ooe.Minecraft.DimensionRequest request) {
      return blockingUnaryCall(
          getChannel(), getSetActiveDimensionMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Force-load/generate chunks in the active dimension within a block-area
     * </pre>
     */
    public com.google.protobuf.Empty preloadChunks(dk.itu.real.ooe.Minecraft.PreloadChunksRequest request) {
      return blockingUnaryCall(
          getChannel(), getPreloadChunksMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   **
   *The main service.
   * </pre>
   */
  public static final class MinecraftServiceFutureStub extends io.grpc.stub.AbstractStub<MinecraftServiceFutureStub> {
    private MinecraftServiceFutureStub(io.grpc.Channel channel) {
      super(channel);
    }

    private MinecraftServiceFutureStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected MinecraftServiceFutureStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new MinecraftServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     ** Spawn multiple blocks. 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> spawnBlocks(
        dk.itu.real.ooe.Minecraft.Blocks request) {
      return futureUnaryCall(
          getChannel().newCall(getSpawnBlocksMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Reads multiple entities *
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Entities> readEntities(
        dk.itu.real.ooe.Minecraft.Uuids request) {
      return futureUnaryCall(
          getChannel().newCall(getReadEntitiesMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Spawn multiple entities. 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Uuids> spawnEntities(
        dk.itu.real.ooe.Minecraft.SpawnEntities request) {
      return futureUnaryCall(
          getChannel().newCall(getSpawnEntitiesMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Return all blocks in a cube 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Blocks> readCube(
        dk.itu.real.ooe.Minecraft.Cube request) {
      return futureUnaryCall(
          getChannel().newCall(getReadCubeMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Fill a cube with a block type 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> fillCube(
        dk.itu.real.ooe.Minecraft.FillCubeRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getFillCubeMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Entities> readEntitiesInSphere(
        dk.itu.real.ooe.Minecraft.Sphere request) {
      return futureUnaryCall(
          getChannel().newCall(getReadEntitiesInSphereMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.BiomeResponse> getBiomeAt(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getGetBiomeAtMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Point> setLoc(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getSetLocMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> setRot(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getSetRotMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> setPlayerLocRot(
        dk.itu.real.ooe.Minecraft.LocRotMsg request) {
      return futureUnaryCall(
          getChannel().newCall(getSetPlayerLocRotMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> initDataGen(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getInitDataGenMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Point> getHighestYAt(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getGetHighestYAtMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Point> setLocY(
        dk.itu.real.ooe.Minecraft.Point request) {
      return futureUnaryCall(
          getChannel().newCall(getSetLocYMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Blocks> readCubeAndBiome(
        dk.itu.real.ooe.Minecraft.Cube request) {
      return futureUnaryCall(
          getChannel().newCall(getReadCubeAndBiomeMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Return all blocks in a cube, including per-block metadata 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.Blocks> readCubeAndBiomeMetadata(
        dk.itu.real.ooe.Minecraft.Cube request) {
      return futureUnaryCall(
          getChannel().newCall(getReadCubeAndBiomeMetadataMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + biome ids, plus sparse metadata for relevant blocks 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.DenseCube> readDenseCubeWithMetadata(
        dk.itu.real.ooe.Minecraft.Cube request) {
      return futureUnaryCall(
          getChannel().newCall(getReadDenseCubeWithMetadataMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     ** Return dense arrays for blocks + a single majority-biome label for the whole cube 
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.DenseCubeMajority> readDenseCubeWithMajority(
        dk.itu.real.ooe.Minecraft.Cube request) {
      return futureUnaryCall(
          getChannel().newCall(getReadDenseCubeWithMajorityMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.TriggerResponse> setModelArea(
        dk.itu.real.ooe.Minecraft.SetAreaRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSetModelAreaMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.BuildingArea> getCurrentBuildingArea(
        com.google.protobuf.Empty request) {
      return futureUnaryCall(
          getChannel().newCall(getGetCurrentBuildingAreaMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.TriggerResponse> triggerModel(
        com.google.protobuf.Empty request) {
      return futureUnaryCall(
          getChannel().newCall(getTriggerModelMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.TriggerResponse> handleLoadPreset(
        dk.itu.real.ooe.Minecraft.LoadPresetRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getHandleLoadPresetMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.TriggerResponse> modifyTerrain(
        dk.itu.real.ooe.Minecraft.TerrainModificationRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getModifyTerrainMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Trigger diffusion-based terrain generation and spawning
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<dk.itu.real.ooe.Minecraft.TriggerResponse> terraform(
        dk.itu.real.ooe.Minecraft.TerraformRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getTerraformMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Switch active dimension/world for subsequent RPC operations; optionally teleport player
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> setActiveDimension(
        dk.itu.real.ooe.Minecraft.DimensionRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSetActiveDimensionMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Force-load/generate chunks in the active dimension within a block-area
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.google.protobuf.Empty> preloadChunks(
        dk.itu.real.ooe.Minecraft.PreloadChunksRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getPreloadChunksMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_SPAWN_BLOCKS = 0;
  private static final int METHODID_READ_ENTITIES = 1;
  private static final int METHODID_SPAWN_ENTITIES = 2;
  private static final int METHODID_READ_CUBE = 3;
  private static final int METHODID_FILL_CUBE = 4;
  private static final int METHODID_READ_ENTITIES_IN_SPHERE = 5;
  private static final int METHODID_GET_BIOME_AT = 6;
  private static final int METHODID_SET_LOC = 7;
  private static final int METHODID_SET_ROT = 8;
  private static final int METHODID_SET_PLAYER_LOC_ROT = 9;
  private static final int METHODID_INIT_DATA_GEN = 10;
  private static final int METHODID_GET_HIGHEST_YAT = 11;
  private static final int METHODID_SET_LOC_Y = 12;
  private static final int METHODID_READ_CUBE_AND_BIOME = 13;
  private static final int METHODID_READ_CUBE_AND_BIOME_METADATA = 14;
  private static final int METHODID_READ_DENSE_CUBE_WITH_METADATA = 15;
  private static final int METHODID_READ_DENSE_CUBE_WITH_MAJORITY = 16;
  private static final int METHODID_SET_MODEL_AREA = 17;
  private static final int METHODID_GET_CURRENT_BUILDING_AREA = 18;
  private static final int METHODID_TRIGGER_MODEL = 19;
  private static final int METHODID_HANDLE_LOAD_PRESET = 20;
  private static final int METHODID_MODIFY_TERRAIN = 21;
  private static final int METHODID_TERRAFORM = 22;
  private static final int METHODID_SET_ACTIVE_DIMENSION = 23;
  private static final int METHODID_PRELOAD_CHUNKS = 24;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final MinecraftServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(MinecraftServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_SPAWN_BLOCKS:
          serviceImpl.spawnBlocks((dk.itu.real.ooe.Minecraft.Blocks) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_READ_ENTITIES:
          serviceImpl.readEntities((dk.itu.real.ooe.Minecraft.Uuids) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities>) responseObserver);
          break;
        case METHODID_SPAWN_ENTITIES:
          serviceImpl.spawnEntities((dk.itu.real.ooe.Minecraft.SpawnEntities) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Uuids>) responseObserver);
          break;
        case METHODID_READ_CUBE:
          serviceImpl.readCube((dk.itu.real.ooe.Minecraft.Cube) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks>) responseObserver);
          break;
        case METHODID_FILL_CUBE:
          serviceImpl.fillCube((dk.itu.real.ooe.Minecraft.FillCubeRequest) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_READ_ENTITIES_IN_SPHERE:
          serviceImpl.readEntitiesInSphere((dk.itu.real.ooe.Minecraft.Sphere) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Entities>) responseObserver);
          break;
        case METHODID_GET_BIOME_AT:
          serviceImpl.getBiomeAt((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BiomeResponse>) responseObserver);
          break;
        case METHODID_SET_LOC:
          serviceImpl.setLoc((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point>) responseObserver);
          break;
        case METHODID_SET_ROT:
          serviceImpl.setRot((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_SET_PLAYER_LOC_ROT:
          serviceImpl.setPlayerLocRot((dk.itu.real.ooe.Minecraft.LocRotMsg) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_INIT_DATA_GEN:
          serviceImpl.initDataGen((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_GET_HIGHEST_YAT:
          serviceImpl.getHighestYAt((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point>) responseObserver);
          break;
        case METHODID_SET_LOC_Y:
          serviceImpl.setLocY((dk.itu.real.ooe.Minecraft.Point) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Point>) responseObserver);
          break;
        case METHODID_READ_CUBE_AND_BIOME:
          serviceImpl.readCubeAndBiome((dk.itu.real.ooe.Minecraft.Cube) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks>) responseObserver);
          break;
        case METHODID_READ_CUBE_AND_BIOME_METADATA:
          serviceImpl.readCubeAndBiomeMetadata((dk.itu.real.ooe.Minecraft.Cube) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.Blocks>) responseObserver);
          break;
        case METHODID_READ_DENSE_CUBE_WITH_METADATA:
          serviceImpl.readDenseCubeWithMetadata((dk.itu.real.ooe.Minecraft.Cube) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCube>) responseObserver);
          break;
        case METHODID_READ_DENSE_CUBE_WITH_MAJORITY:
          serviceImpl.readDenseCubeWithMajority((dk.itu.real.ooe.Minecraft.Cube) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.DenseCubeMajority>) responseObserver);
          break;
        case METHODID_SET_MODEL_AREA:
          serviceImpl.setModelArea((dk.itu.real.ooe.Minecraft.SetAreaRequest) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse>) responseObserver);
          break;
        case METHODID_GET_CURRENT_BUILDING_AREA:
          serviceImpl.getCurrentBuildingArea((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.BuildingArea>) responseObserver);
          break;
        case METHODID_TRIGGER_MODEL:
          serviceImpl.triggerModel((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse>) responseObserver);
          break;
        case METHODID_HANDLE_LOAD_PRESET:
          serviceImpl.handleLoadPreset((dk.itu.real.ooe.Minecraft.LoadPresetRequest) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse>) responseObserver);
          break;
        case METHODID_MODIFY_TERRAIN:
          serviceImpl.modifyTerrain((dk.itu.real.ooe.Minecraft.TerrainModificationRequest) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse>) responseObserver);
          break;
        case METHODID_TERRAFORM:
          serviceImpl.terraform((dk.itu.real.ooe.Minecraft.TerraformRequest) request,
              (io.grpc.stub.StreamObserver<dk.itu.real.ooe.Minecraft.TriggerResponse>) responseObserver);
          break;
        case METHODID_SET_ACTIVE_DIMENSION:
          serviceImpl.setActiveDimension((dk.itu.real.ooe.Minecraft.DimensionRequest) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
          break;
        case METHODID_PRELOAD_CHUNKS:
          serviceImpl.preloadChunks((dk.itu.real.ooe.Minecraft.PreloadChunksRequest) request,
              (io.grpc.stub.StreamObserver<com.google.protobuf.Empty>) responseObserver);
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

  private static abstract class MinecraftServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    MinecraftServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return dk.itu.real.ooe.Minecraft.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("MinecraftService");
    }
  }

  private static final class MinecraftServiceFileDescriptorSupplier
      extends MinecraftServiceBaseDescriptorSupplier {
    MinecraftServiceFileDescriptorSupplier() {}
  }

  private static final class MinecraftServiceMethodDescriptorSupplier
      extends MinecraftServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    MinecraftServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (MinecraftServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new MinecraftServiceFileDescriptorSupplier())
              .addMethod(getSpawnBlocksMethod())
              .addMethod(getReadEntitiesMethod())
              .addMethod(getSpawnEntitiesMethod())
              .addMethod(getReadCubeMethod())
              .addMethod(getFillCubeMethod())
              .addMethod(getReadEntitiesInSphereMethod())
              .addMethod(getGetBiomeAtMethod())
              .addMethod(getSetLocMethod())
              .addMethod(getSetRotMethod())
              .addMethod(getSetPlayerLocRotMethod())
              .addMethod(getInitDataGenMethod())
              .addMethod(getGetHighestYAtMethod())
              .addMethod(getSetLocYMethod())
              .addMethod(getReadCubeAndBiomeMethod())
              .addMethod(getReadCubeAndBiomeMetadataMethod())
              .addMethod(getReadDenseCubeWithMetadataMethod())
              .addMethod(getReadDenseCubeWithMajorityMethod())
              .addMethod(getSetModelAreaMethod())
              .addMethod(getGetCurrentBuildingAreaMethod())
              .addMethod(getTriggerModelMethod())
              .addMethod(getHandleLoadPresetMethod())
              .addMethod(getModifyTerrainMethod())
              .addMethod(getTerraformMethod())
              .addMethod(getSetActiveDimensionMethod())
              .addMethod(getPreloadChunksMethod())
              .build();
        }
      }
    }
    return result;
  }
}
