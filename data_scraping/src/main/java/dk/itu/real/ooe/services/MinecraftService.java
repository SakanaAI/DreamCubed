package dk.itu.real.ooe.services;

import com.google.protobuf.Empty;
import dk.itu.real.ooe.Minecraft;
import dk.itu.real.ooe.Minecraft.*;
import dk.itu.real.ooe.MinecraftServiceGrpc.MinecraftServiceImplBase;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.spongepowered.api.Game;
import org.spongepowered.api.Sponge;
import org.spongepowered.api.entity.Entity;
import org.spongepowered.api.entity.EntityTypes;
import org.spongepowered.api.entity.living.animal.Chicken;
import org.spongepowered.api.entity.living.player.Player;
import org.spongepowered.api.entity.living.player.gamemode.GameModes;
import org.spongepowered.api.event.cause.EventContextKeys;
import org.spongepowered.api.event.CauseStackManager.StackFrame;
import org.spongepowered.api.event.cause.entity.spawn.SpawnTypes;
import org.spongepowered.api.block.BlockState;
import org.spongepowered.api.block.BlockType;
import org.spongepowered.api.block.BlockTypes;
import org.spongepowered.api.data.key.Keys;
import org.spongepowered.api.data.manipulator.mutable.block.DirectionalData;
import org.spongepowered.api.data.type.SlabType;
import org.spongepowered.api.data.type.StairShape;
import org.spongepowered.api.block.trait.BlockTrait;
import org.spongepowered.api.plugin.PluginContainer;
import org.spongepowered.api.scheduler.Task;
import org.spongepowered.api.text.Text;
import org.spongepowered.api.util.Direction;
import org.spongepowered.api.util.rotation.Rotation;
import org.spongepowered.api.world.Location;
import org.spongepowered.api.world.World;
import org.spongepowered.api.world.DimensionType;
import org.spongepowered.api.world.DimensionTypes;
import org.spongepowered.api.world.weather.Weathers;
// import GameRule
import org.spongepowered.api.world.gamerule.DefaultGameRules;

import com.flowpowered.math.vector.Vector3d;
import com.flowpowered.math.vector.Vector3i;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import org.spongepowered.api.command.CommandResult;
import org.spongepowered.api.command.CommandSource;
import org.spongepowered.api.command.args.CommandContext;
import org.spongepowered.api.command.args.GenericArguments;
import org.spongepowered.api.command.spec.CommandSpec;
import org.spongepowered.api.effect.particle.ParticleEffect;
import org.spongepowered.api.effect.particle.ParticleTypes;
import org.spongepowered.api.event.Listener;
import org.spongepowered.api.event.game.state.GameStartedServerEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import dk.itu.real.ooe.MinecraftServiceGrpc;
import org.spongepowered.api.event.block.InteractBlockEvent;
import dk.itu.real.ooe.Minecraft.TerrainModificationRequest;
// import dk.itu.real.ooe.Minecraft.ModificationRegion;
import dk.itu.real.ooe.Minecraft.ModificationType;
import dk.itu.real.ooe.Minecraft.RegionModification;

public class MinecraftService extends MinecraftServiceImplBase {

    private static final Logger logger = LoggerFactory.getLogger(MinecraftService.class);

    // variables for interactive tool and visualization
    private static final int LATENT_SIZE = 6;
    private static final int OUTPUT_SIZE = 24;
    private static final int GAP_BETWEEN_LATENT = 2;
    private static final int GAP_BEFORE_OUTPUT = 10;

    private final PluginContainer plugin;
    // private final Game game;
    private final Map<String, String> blockNamesToBlockTypes = new HashMap<>(); // minecraft:dirt --> DIRT
    private final Map<String, dk.itu.real.ooe.Minecraft.BlockType> blockNameToEnum = new HashMap<>(); // minecraft:dirt -> enum
    private final Map<String, Byte> blockNameToIdByte = new HashMap<>(); // minecraft:dirt -> proto id (uint8)
    private final Map<String, String> entityNamesToEntityTypes = new HashMap<>(); //minecraft:creeper --> CREEPER

    private Task particleTask = null;  // Store reference to running particle task
    private BuildingArea currentBuildingArea = null;  // Store current area coordinates
    private Task terraformParticleTask = null; // Outline for /terraform

    private TerrainTool terrainTool;
    private volatile World activeWorldOverride = null;

    public MinecraftService(PluginContainer plugin) throws IllegalAccessException {
        this.plugin = plugin;
        this.terrainTool = new TerrainTool(plugin);
        // this.game = game;
        registerCommands();
        
        // Add this line to register the event listeners
        Sponge.getEventManager().registerListeners(plugin, this);

        for (Field field : BlockTypes.class.getFields()) {
            BlockType blockType = (BlockType) field.get(null);
            String key = blockType.getName();
            String value = field.getName();
            blockNamesToBlockTypes.put(key, value);
            try {
                dk.itu.real.ooe.Minecraft.BlockType enumConst = dk.itu.real.ooe.Minecraft.BlockType.valueOf(value);
                blockNameToEnum.put(key, enumConst);
                blockNameToIdByte.put(key, (byte)(enumConst.getNumber() & 0xFF));
            } catch (Exception ignored) {}
        }

        for (Field field : EntityTypes.class.getFields()) {
            org.spongepowered.api.entity.EntityType entityType = (org.spongepowered.api.entity.EntityType) field.get(null);
            String key = entityType.getName();
            String value = field.getName();
            entityNamesToEntityTypes.put(key, value);
        }
    }

    private void preloadChunksForWindow(World world, Vector3d min, Vector3d max) {
        try {
            int cminX = ((int)Math.floor(min.getX())) >> 4;
            int cmaxX = ((int)Math.floor(max.getX())) >> 4;
            int cminZ = ((int)Math.floor(min.getZ())) >> 4;
            int cmaxZ = ((int)Math.floor(max.getZ())) >> 4;
            for (int cx = cminX; cx <= cmaxX; cx++) {
                for (int cz = cminZ; cz <= cmaxZ; cz++) {
                    try {
                        // Prefer vector overload if available
                        world.loadChunk(new Vector3i(cx, 0, cz), true);
                    } catch (Throwable t) {
                        try {
                            // Fallback to primitive overload with Y=0 (Sponge API expects x,y,z chunk coords)
                            world.loadChunk(cx, 0, cz, true);
                        } catch (Throwable ignored) {}
                    }
                }
            }
        } catch (Exception ignored) {}
    }

    private Player getConfiguredPlayer() {
        String playerName = System.getProperty("MCRPC_PLAYER");
        if (playerName == null || playerName.trim().isEmpty()) {
            playerName = System.getenv("MCRPC_PLAYER");
        }
        if (playerName != null && !playerName.trim().isEmpty()) {
            Optional<Player> player = Sponge.getServer().getPlayer(playerName.trim());
            if (player.isPresent()) {
                return player.get();
            }
            throw new IllegalStateException("Configured Minecraft player is not online: " + playerName);
        }

        Optional<Player> firstPlayer = Sponge.getServer().getOnlinePlayers().stream().findFirst();
        if (firstPlayer.isPresent()) {
            return firstPlayer.get();
        }
        throw new IllegalStateException("No online Minecraft player is available for player RPC calls.");
    }

    /**
     * Resolve the target world based on an optional environment/system property MCRPC_DIM.
     * Supported values: "overworld", "nether", "the_end".
     * Falls back to the server's first loaded world if no match found.
     */
    private World resolveWorld() {
        try {
            if (activeWorldOverride != null) {
                return activeWorldOverride;
            }
            String dim = System.getProperty("MCRPC_DIM");
            if (dim == null || dim.isEmpty()) {
                dim = System.getenv("MCRPC_DIM");
            }
            if (dim != null) {
                dim = dim.trim().toLowerCase(java.util.Locale.ROOT);
            }

            if ("nether".equals(dim) || "the_nether".equals(dim)) {
                World w = findWorldByDimensionType(DimensionTypes.NETHER);
                if (w != null) return w;
                java.util.Optional<World> ow = Sponge.getServer().getWorld("DIM-1");
                if (ow.isPresent()) return ow.get();
                ow = Sponge.getServer().getWorld("world_nether");
                if (ow.isPresent()) return ow.get();
                // Common SpongeVanilla pattern: <level-name>_nether
                for (World cand : Sponge.getServer().getWorlds()) {
                    if (cand.getName().toLowerCase(java.util.Locale.ROOT).endsWith("_nether")) {
                        return cand;
                    }
                }
            } else if ("the_end".equals(dim) || "end".equals(dim)) {
                World w = findWorldByDimensionType(DimensionTypes.THE_END);
                if (w != null) return w;
                java.util.Optional<World> ow = Sponge.getServer().getWorld("DIM1");
                if (ow.isPresent()) return ow.get();
                for (World cand : Sponge.getServer().getWorlds()) {
                    if (cand.getName().toLowerCase(java.util.Locale.ROOT).endsWith("_the_end") ||
                        cand.getName().equalsIgnoreCase("the_end")) {
                        return cand;
                    }
                }
            } else if ("overworld".equals(dim) || "world".equals(dim)) {
                World w = findWorldByDimensionType(DimensionTypes.OVERWORLD);
                if (w != null) return w;
            }
        } catch (Throwable ignored) {}
        return Sponge.getServer().getWorlds().iterator().next();
    }

    private World findWorldByDimensionType(DimensionType type) {
        try {
            for (World w : Sponge.getServer().getWorlds()) {
                try {
                    if (w.getDimension().getType().equals(type)) {
                        return w;
                    }
                } catch (Throwable t) {
                    try {
                        java.lang.reflect.Method m = w.getClass().getMethod("getDimensionType");
                        Object dt = m.invoke(w);
                        if (type.equals(dt)) {
                            return w;
                        }
                    } catch (Throwable ignored) {}
                }
            }
        } catch (Throwable ignored) {}
        return null;
    }

    @Override
    public void preloadChunks(Minecraft.PreloadChunksRequest request, StreamObserver<Empty> responseObserver) {
        Task.builder().execute(() -> {
            World world = resolveWorld();
            int minX = Math.min(request.getMinX(), request.getMaxX());
            int maxX = Math.max(request.getMinX(), request.getMaxX());
            int minZ = Math.min(request.getMinZ(), request.getMaxZ());
            int maxZ = Math.max(request.getMinZ(), request.getMaxZ());
            int cminX = (int)Math.floor(minX / 16.0);
            int cmaxX = (int)Math.floor(maxX / 16.0);
            int cminZ = (int)Math.floor(minZ / 16.0);
            int cmaxZ = (int)Math.floor(maxZ / 16.0);
            for (int cx = cminX; cx <= cmaxX; cx++) {
                for (int cz = cminZ; cz <= cmaxZ; cz++) {
                    try {
                        try {
                            world.loadChunk(new Vector3i(cx, 0, cz), true);
                        } catch (Throwable t) {
                            world.loadChunk(cx, 0, cz, true);
                        }
                    } catch (Throwable ignored) {}
                }
            }
            responseObserver.onNext(Empty.getDefaultInstance());
            responseObserver.onCompleted();
        }).name("preloadChunks").submit(plugin);
    }

    @Override
    public void setActiveDimension(Minecraft.DimensionRequest request, StreamObserver<Empty> responseObserver) {
        Task.builder().execute(() -> {
            World target = null;
            try {
                String dim = request.getDimension();
                if (dim != null) {
                    dim = dim.trim().toLowerCase(java.util.Locale.ROOT);
                }
                if ("nether".equals(dim) || "the_nether".equals(dim)) {
                    target = findWorldByDimensionType(DimensionTypes.NETHER);
                    if (target == null) {
                        // Try load by common name
                        java.util.Optional<World> ow = Sponge.getServer().getWorld("DIM-1");
                        if (ow.isPresent()) {
                            target = ow.get();
                        } else {
                            // Try load world properties and load the world explicitly
                            java.util.Optional<org.spongepowered.api.world.storage.WorldProperties> props =
                                Sponge.getServer().getWorldProperties("DIM-1");
                            if (props.isPresent()) {
                                org.spongepowered.api.world.storage.WorldProperties p = props.get();
                                try {
                                    p.setKeepSpawnLoaded(true);
                                } catch (Throwable ignored) {}
                                java.util.Optional<World> loaded = Sponge.getServer().loadWorld(p);
                                if (loaded.isPresent()) {
                                    target = loaded.get();
                                }
                            } else {
                                // Last resort: scan all world properties for NETHER type and load the first match
                                try {
                                    for (org.spongepowered.api.world.storage.WorldProperties wp : Sponge.getServer().getAllWorldProperties()) {
                                        try {
                                            if (wp.getDimensionType().equals(DimensionTypes.NETHER)) {
                                                try {
                                                    wp.setKeepSpawnLoaded(true);
                                                } catch (Throwable ignored) {}
                                                java.util.Optional<World> loaded = Sponge.getServer().loadWorld(wp);
                                                if (loaded.isPresent()) {
                                                    target = loaded.get();
                                                    break;
                                                }
                                            }
                                        } catch (Throwable ignored) {}
                                    }
                                } catch (Throwable ignored) {}
                            }
                        }
                    }
                } else if ("the_end".equals(dim) || "end".equals(dim)) {
                    target = findWorldByDimensionType(DimensionTypes.THE_END);
                    if (target == null) {
                        java.util.Optional<World> ow = Sponge.getServer().getWorld("DIM1");
                        if (ow.isPresent()) {
                            target = ow.get();
                        } else {
                            java.util.Optional<org.spongepowered.api.world.storage.WorldProperties> props =
                                Sponge.getServer().getWorldProperties("DIM1");
                            if (props.isPresent()) {
                                org.spongepowered.api.world.storage.WorldProperties p = props.get();
                                try {
                                    p.setKeepSpawnLoaded(true);
                                } catch (Throwable ignored) {}
                                java.util.Optional<World> loaded = Sponge.getServer().loadWorld(p);
                                if (loaded.isPresent()) {
                                    target = loaded.get();
                                }
                            }
                        }
                    }
                } else if ("overworld".equals(dim) || "world".equals(dim)) {
                    target = findWorldByDimensionType(DimensionTypes.OVERWORLD);
                }
                if (target == null) {
                    // Fallback: keep current world to avoid null
                    target = Sponge.getServer().getWorlds().iterator().next();
                }
                activeWorldOverride = target;

                if (request.getTeleportPlayer()) {
                    try {
                        Player player = getConfiguredPlayer();
                        int tx = request.getX();
                        int tz = request.getZ();
                        int ty = request.getY();
                        if (tx == 0 && tz == 0 && ty == 0) {
                            // default safe spot near 0,0
                            tx = 0;
                            tz = 0;
                            int hy = target.getHighestYAt(tx, tz);
                            ty = (hy > 0) ? hy : 64;
                        } else if (ty <= 0) {
                            int hy = target.getHighestYAt(tx, tz);
                            ty = (hy > 0) ? hy : 64;
                        }
                        player.setLocation(new Location<>(target, tx, ty, tz));
                    } catch (Exception ignored) {}
                }
                responseObserver.onNext(Empty.getDefaultInstance());
                responseObserver.onCompleted();
            } catch (Exception e) {
                responseObserver.onNext(Empty.getDefaultInstance());
                responseObserver.onCompleted();
            }
        }).name("setActiveDimension").submit(plugin);
    }

    @Override
    public void setModelArea(SetAreaRequest request, StreamObserver<TriggerResponse> responseObserver) {
        Task.builder().execute(() -> {
            World world = resolveWorld();
            Point origin = request.getOrigin();
            
            // Clear any existing areas first
            clearPreviousAreas();
            
            if (request.getUseParticles()) {
                // Schedule repeating task for particle effects
                particleTask = Task.builder()
                    .interval(500, TimeUnit.MILLISECONDS)  // Refresh every 0.5 seconds
                    .execute(() -> {
                        // First latent cube
                        drawParticleCube(world, origin, LATENT_SIZE);
                        
                        // Second latent cube
                        Point secondCube = Point.newBuilder()
                            .setX(origin.getX() + LATENT_SIZE + GAP_BETWEEN_LATENT)
                            .setY(origin.getY())
                            .setZ(origin.getZ())
                            .build();
                        drawParticleCube(world, secondCube, LATENT_SIZE);
                        
                        // Output cube
                        Point outputCube = Point.newBuilder()
                            .setX(origin.getX() + 2 * LATENT_SIZE + GAP_BETWEEN_LATENT + GAP_BEFORE_OUTPUT)
                            .setY(origin.getY())
                            .setZ(origin.getZ())
                            .build();
                        drawParticleCube(world, outputCube, OUTPUT_SIZE);
                    })
                    .submit(plugin);
            } else {
                // Glass shell approach
                spawnGlassShell(world, origin, LATENT_SIZE);
                
                Point secondCube = Point.newBuilder()
                    .setX(origin.getX() + LATENT_SIZE + GAP_BETWEEN_LATENT)
                    .setY(origin.getY())
                    .setZ(origin.getZ())
                    .build();
                spawnGlassShell(world, secondCube, LATENT_SIZE);
                
                Point outputCube = Point.newBuilder()
                    .setX(origin.getX() + 2 * LATENT_SIZE + GAP_BETWEEN_LATENT + GAP_BEFORE_OUTPUT)
                    .setY(origin.getY())
                    .setZ(origin.getZ())
                    .build();
                spawnGlassShell(world, outputCube, OUTPUT_SIZE);
            }
            
            // Store the areas for later use
            currentBuildingArea = BuildingArea.newBuilder()
                .setFirstLatentOrigin(origin)
                .setSecondLatentOrigin(Point.newBuilder()
                    .setX(origin.getX() + LATENT_SIZE + GAP_BETWEEN_LATENT)
                    .setY(origin.getY())
                    .setZ(origin.getZ()))
                .setOutputOrigin(Point.newBuilder()
                    .setX(origin.getX() + 2 * LATENT_SIZE + GAP_BETWEEN_LATENT + GAP_BEFORE_OUTPUT)
                    .setY(origin.getY())
                    .setZ(origin.getZ()))
                .build();
            
            // Update terrain tool with new building area
            terrainTool.setBuildingArea(currentBuildingArea);
            
            responseObserver.onNext(TriggerResponse.newBuilder().setSuccess(true).build());
            responseObserver.onCompleted();
        }).name("setModelArea").submit(plugin);
    }

    private void drawParticleCube(World world, Point origin, int size) {
        // Add 1 block padding on each side
        int startX = origin.getX() - 1;
        int startY = origin.getY() - 1;
        int startZ = origin.getZ() - 1;
        int endX = startX + size + 1;
        int endY = startY + size + 1;
        int endZ = startZ + size + 1;
        
        // Draw all 12 edges of the cube
        for (double i = 0; i <= size + 1; i += 0.5) {  // Smaller step size for denser particles
            // Bottom edges
            spawnParticle(world, startX + i, startY, startZ);        // Front edge
            spawnParticle(world, startX + i, startY, endZ);         // Back edge
            spawnParticle(world, startX, startY, startZ + i);       // Left edge
            spawnParticle(world, endX, startY, startZ + i);         // Right edge
            
            // Top edges
            spawnParticle(world, startX + i, endY, startZ);        // Front edge
            spawnParticle(world, startX + i, endY, endZ);         // Back edge
            spawnParticle(world, startX, endY, startZ + i);       // Left edge
            spawnParticle(world, endX, endY, startZ + i);         // Right edge
            
            // Vertical edges
            spawnParticle(world, startX, startY + i, startZ);     // Front-left
            spawnParticle(world, endX, startY + i, startZ);      // Front-right
            spawnParticle(world, startX, startY + i, endZ);      // Back-left
            spawnParticle(world, endX, startY + i, endZ);       // Back-right
        }
    }

    private void startTerraformOutline(Point origin, int size) {
        stopTerraformOutline();
        terraformParticleTask = Task.builder()
            .interval(500, TimeUnit.MILLISECONDS)
            .execute(() -> {
                World world = resolveWorld();
                drawParticleCube(world, origin, size);
            })
            .submit(plugin);
    }

    private void stopTerraformOutline() {
        if (terraformParticleTask != null) {
            terraformParticleTask.cancel();
            terraformParticleTask = null;
        }
    }

    private void spawnParticle(World world, double x, double y, double z) {
        world.spawnParticles(
            ParticleEffect.builder().type(ParticleTypes.END_ROD).build(),
            new Vector3d(x + 0.5, y + 0.5, z + 0.5)
        );
    }

    private void spawnGlassShell(World world, Point origin, int size) {
        // Spawn glass blocks to form a shell
        for (int x = 0; x <= size + 1; x++) {
            for (int y = 0; y <= size + 1; y++) {
                for (int z = 0; z <= size + 1; z++) {
                    // Only place glass if it's on the shell (not inside)
                    if (x == 0 || x == size + 1 || 
                        y == 0 || y == size + 1 || 
                        z == 0 || z == size + 1) {
                        world.getLocation(
                            origin.getX() + x - 1,
                            origin.getY() + y - 1,
                            origin.getZ() + z - 1
                        ).setBlockType(BlockTypes.GLASS);
                    }
                }
            }
        }
    }

    @Override
    public void spawnBlocks(Blocks request, StreamObserver<Empty> responseObserver) {
        Task.builder().execute(() -> {
                    World world = resolveWorld();
                    for (Block block : request.getBlocksList()) {
                        try {
                            BlockType blockType = (BlockType) BlockTypes.class.getField(block.getType().toString()).get(null);
                            Point pos = block.getPosition();
                            Orientation orientation = block.getOrientation();
                            world.setBlockType(pos.getX(), pos.getY(), pos.getZ(), blockType);
                            if (blockType.getDefaultState().supports(Keys.DIRECTION)) {
                                setOrientation(world.getLocation(pos.getX(), pos.getY(), pos.getZ()), orientation, blockType);
                            }
                        } catch (IllegalStateException | NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e ){
                            this.plugin.getLogger().info(e.getClass().getCanonicalName());
                            this.plugin.getLogger().info(e.getMessage());
                        }
                    }

                    responseObserver.onNext(Empty.getDefaultInstance());
                    responseObserver.onCompleted();
                }
        ).name("spawnBlocks").submit(plugin);
    }

    @Override
    public void readEntitiesInSphere(Sphere request, StreamObserver<Entities> responseObserver) {
        Task.builder().execute(() -> {
                Entities.Builder builder = Entities.newBuilder();
                World world = resolveWorld();
                ArrayList<Entity> entities = (ArrayList<Entity>) world.getNearbyEntities(new Vector3d(request.getCenter().getX(), request.getCenter().getY(), request.getCenter().getZ()), request.getRadius());
                for (Entity entity : entities) {
	                builder.addEntities(Minecraft.Entity.newBuilder()
                            .setId(entity.getUniqueId().toString())
                            .setType(Minecraft.EntityType.valueOf("ENTITY_" + entityNamesToEntityTypes.get(entity.getType().getName())))
                            .setPosition(Point.newBuilder()
                                            .setX((int)entity.getLocation().getX())
                                            .setY((int)entity.getLocation().getY())
                                            .setZ((int)entity.getLocation().getZ())
                                            .build())
                            .setIsLoaded(entity.isLoaded()))
                            .build();
                }
                responseObserver.onNext(builder.build());
                responseObserver.onCompleted();
            }
        ).name("readCube").submit(plugin);
    }

    @Override
    public void readEntities(Uuids request, StreamObserver<Entities> responseObserver) {
        Task.builder().execute(() -> {
            Entities.Builder builder = Entities.newBuilder();
            World world = resolveWorld();
            for(String id : request.getUuidsList()) {
                Optional<Entity> entityOption = world.getEntity(UUID.fromString(id));
                if(!entityOption.isPresent()){
                    builder.addEntities(Minecraft.Entity.newBuilder()
                        //Proto ignores defualt values so there is no need to set type, position and isloaded
                        .setId(id)).build();
                } else {
                    org.spongepowered.api.entity.Entity entity = entityOption.get();
                    Location location = entity.getLocation();
                    builder.addEntities(Minecraft.Entity.newBuilder()
                        .setId(id)
                        .setType(Minecraft.EntityType.valueOf("ENTITY_" + entityNamesToEntityTypes.get(entity.getType().getName())))
                        .setPosition(Point.newBuilder()
                                        .setX((int)location.getX())
                                        .setY((int)location.getY())
                                        .setZ((int)location.getZ())
                                        .build())
                        .setIsLoaded(entity.isLoaded())
                        ).build();
                }
            }
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("spawnEntities").submit(plugin);
    }

    @Override
    public void spawnEntities(SpawnEntities request, StreamObserver<Uuids> responseObserver){
        Task.builder().execute(() -> {
            Uuids.Builder builder = Uuids.newBuilder();
            World world = resolveWorld();
            for (dk.itu.real.ooe.Minecraft.SpawnEntity entity : request.getSpawnEntitiesList()) {
                try {
                    org.spongepowered.api.entity.EntityType entityType = (org.spongepowered.api.entity.EntityType) EntityTypes.class.getField(entity.getType().toString().split("_", 2)[1]).get(null);
                    Point pos = entity.getSpawnPosition();
                    org.spongepowered.api.entity.Entity newEntity = world.createEntity(entityType, new Vector3d(pos.getX(), pos.getY(), pos.getZ()));
                    try (StackFrame frame = Sponge.getCauseStackManager().pushCauseFrame()) {
                        frame.addContext(EventContextKeys.SPAWN_TYPE, SpawnTypes.PLUGIN);
                        world.spawnEntity(newEntity);
                    }
                builder.addUuids(newEntity.getUniqueId().toString()).build();
                } catch (IllegalStateException | NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e){
                    this.plugin.getLogger().info(e.getMessage());
                }
            }
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("spawnEntities").submit(plugin);
    }

    @Override
    public void readCube(Cube cube, StreamObserver<Blocks> responseObserver) {
        Task.builder().execute(() -> {
            Blocks.Builder builder = Blocks.newBuilder();
            World world = resolveWorld();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(cube.getMin(), cube.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();

            for (int x = (int)min.getX(); x <= max.getX(); x++) {
                for (int y = (int)min.getY(); y <= max.getY(); y++) {
                    for (int z = (int)min.getZ(); z <= max.getZ(); z++) {
                        String name = world.getLocation(x, y, z).getBlock().getType().getName();
                        builder.addBlocks(Block.newBuilder()
                                .setPosition(Point.newBuilder()
                                                .setX(x)
                                                .setY(y)
                                                .setZ(z)
                                                .build())
                                .setType(Minecraft.BlockType.valueOf(blockNamesToBlockTypes.get(name))).build());
                    }
                }
            }
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("readCube").submit(plugin);
    }

    @Override
    public void readCubeAndBiome(Cube cube, StreamObserver<Blocks> responseObserver) {
        Task.builder().execute(() -> {
            Blocks.Builder builder = Blocks.newBuilder();
            World world = resolveWorld();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(cube.getMin(), cube.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();

            // Preload intersecting chunks before reading if requested by client
            try {
                if (cube.getPreloadChunks()) {
                    preloadChunksForWindow(world, min, max);
                }
            } catch (Exception ignored) {}

            for (int x = (int)min.getX(); x <= max.getX(); x++) {
                for (int z = (int)min.getZ(); z <= max.getZ(); z++) {
                    // Cache biome per column (y-invariant)
                    String biomeNameColumn = world.getBiome(x, (int)min.getY(), z).getId().split(":")[1];
                    for (int y = (int)min.getY(); y <= max.getY(); y++) {
                        // Avoid Location allocation here
                        String blockName = world.getBlock(x, y, z).getType().getName();
                        String blockNameLower = (blockName == null) ? null : blockName.toLowerCase(java.util.Locale.ROOT);

                        builder.addBlocks(Block.newBuilder()
                                .setPosition(Point.newBuilder()
                                        .setX(x)
                                        .setY(y)
                                        .setZ(z)
                                        .build())
                                .setType(Minecraft.BlockType.valueOf(blockNamesToBlockTypes.get(blockName)))
                                .setBiome(biomeNameColumn)
                                .build());
                    }
                }
            }

            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();

            // Aggressively unload chunks we touched to avoid gradual memory/tick growth
            // try {
            //     int cminX = ((int)min.getX()) >> 4;
            //     int cmaxX = ((int)max.getX()) >> 4;
            //     int cminZ = ((int)min.getZ()) >> 4;
            //     int cmaxZ = ((int)max.getZ()) >> 4;
            //     for (int cx = cminX; cx <= cmaxX; cx++) {
            //         for (int cz = cminZ; cz <= cmaxZ; cz++) {
            //             java.util.Optional<org.spongepowered.api.world.Chunk> ch = world.getChunk(new Vector3i(cx, 0, cz));
            //             if (ch.isPresent()) {
            //                 world.unloadChunk(ch.get());
            //             }
            //         }
            //     }
            // } catch (Exception ignored) {}
        }).name("readCubeAndBiome").submit(plugin);
    }

    @Override
    public void readCubeAndBiomeMetadata(Cube cube, StreamObserver<Blocks> responseObserver) {
        Task.builder().execute(() -> {
            Blocks.Builder builder = Blocks.newBuilder();
            World world = resolveWorld();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(cube.getMin(), cube.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();

            // Preload intersecting chunks before reading if requested by client
            try {
                if (cube.getPreloadChunks()) {
                    preloadChunksForWindow(world, min, max);
                }
            } catch (Exception ignored) {}

            for (int x = (int)min.getX(); x <= max.getX(); x++) {
                for (int z = (int)min.getZ(); z <= max.getZ(); z++) {
                    // Cache biome per column (y-invariant)
                    String biomeNameColumn = world.getBiome(x, (int)min.getY(), z).getId().split(":")[1];
                    for (int y = (int)min.getY(); y <= max.getY(); y++) {
                        // Fetch type without creating Location
                        String blockName = world.getBlock(x, y, z).getType().getName();
                        String blockNameLower = (blockName == null) ? null : blockName.toLowerCase(java.util.Locale.ROOT);

                        Minecraft.Block.Builder blockBuilder = Block.newBuilder()
                                .setPosition(Point.newBuilder()
                                        .setX(x)
                                        .setY(y)
                                        .setZ(z)
                                        .build())
                                .setType(Minecraft.BlockType.valueOf(blockNamesToBlockTypes.get(blockName)))
                                .setBiome(biomeNameColumn);

                        // Collect metadata only for relevant blocks (stairs, slabs)
                        boolean relevant = (blockNameLower != null) && (blockNameLower.endsWith("_stairs") || blockNameLower.contains("slab"));
                        if (relevant) {
                            Location<World> location = world.getLocation(x, y, z);
                            Map<String, String> metadata = extractSlabAndStairMetadata(location);
                            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                                blockBuilder.putMetadata(entry.getKey(), entry.getValue());
                            }
                        }

                        builder.addBlocks(blockBuilder.build());
                    }
                }
            }

            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();

            // Aggressively unload chunks we touched to avoid gradual memory/tick growth
            // try {
            //     int cminX = ((int)min.getX()) >> 4;
            //     int cmaxX = ((int)max.getX()) >> 4;
            //     int cminZ = ((int)min.getZ()) >> 4;
            //     int cmaxZ = ((int)max.getZ()) >> 4;
            //     for (int cx = cminX; cx <= cmaxX; cx++) {
            //         for (int cz = cminZ; cz <= cmaxZ; cz++) {
            //             java.util.Optional<org.spongepowered.api.world.Chunk> ch = world.getChunk(new Vector3i(cx, 0, cz));
            //             if (ch.isPresent()) {
            //                 world.unloadChunk(ch.get());
            //             }
            //         }
            //     }
            // } catch (Exception ignored) {}
        }).name("readCubeAndBiomeMetadata").submit(plugin);
    }

    @Override
    public void readDenseCubeWithMetadata(Cube cube, StreamObserver<Minecraft.DenseCube> responseObserver) {
        Task.builder().execute(() -> {
            World world = resolveWorld();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(cube.getMin(), cube.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();

            // Preload intersecting chunks before reading if requested by client
            try {
                if (cube.getPreloadChunks()) {
                    preloadChunksForWindow(world, min, max);
                }
            } catch (Exception ignored) {}

            int sx = (int)(max.getX() - min.getX() + 1);
            int sy = (int)(max.getY() - min.getY() + 1);
            int sz = (int)(max.getZ() - min.getZ() + 1);
            int total = sx * sy * sz;

            byte[] types = new byte[total];
            byte[] biomeIds = new byte[total];
            java.util.Map<String, Integer> biomeToIndex = new java.util.HashMap<>();
            java.util.List<String> biomePalette = new java.util.ArrayList<>();
            java.util.List<Minecraft.DenseBlockMetadata> metaList = new java.util.ArrayList<>();

            for (int xi = 0; xi < sx; xi++) {
                int x = (int)min.getX() + xi;
                // Cache biome id per column; convert string to a small id placeholder
                String biomeIdStr = world.getBiome(x, (int)min.getY(), (int)min.getZ()).getId();
                // palette index for (x,*,zmin) column id as fallback
                int colIdx;
                if (biomeIdStr == null) {
                    colIdx = 0;
                } else {
                    Integer got = biomeToIndex.get(biomeIdStr);
                    if (got == null) {
                        got = biomePalette.size();
                        biomePalette.add(biomeIdStr);
                        biomeToIndex.put(biomeIdStr, got);
                    }
                    colIdx = got;
                }
                for (int zi = 0; zi < sz; zi++) {
                    int z = (int)min.getZ() + zi;
                    // Prefer column biome at (x, any y, z)
                    String biomeName = world.getBiome(x, (int)min.getY(), z).getId();
                    int biomeIdx;
                    if (biomeName == null) {
                        biomeIdx = colIdx;
                    } else {
                        Integer got2 = biomeToIndex.get(biomeName);
                        if (got2 == null) {
                            got2 = biomePalette.size();
                            biomePalette.add(biomeName);
                            biomeToIndex.put(biomeName, got2);
                        }
                        biomeIdx = got2;
                    }
                    for (int yi = 0; yi < sy; yi++) {
                        int y = (int)min.getY() + yi;
                        int lin = ((xi * sy) + yi) * sz + zi;
                        // Get type without Location first
                        String blockName = world.getBlock(x, y, z).getType().getName();
                        Byte idByte = blockNameToIdByte.get(blockName);
                        types[lin] = (idByte != null) ? idByte.byteValue() : (byte)0;
                        biomeIds[lin] = (byte)(biomeIdx & 0xFF);

                        boolean relevant = false;
                        if (blockName != null) {
                            relevant = blockName.endsWith("_stairs") || blockName.contains("slab");
                        }
                        if (relevant) {
                            Location<World> location = world.getLocation(x, y, z);
                            Map<String, String> md = extractSlabAndStairMetadata(location);
                            if (md != null && !md.isEmpty()) {
                                Minecraft.DenseBlockMetadata.Builder mBuilder = Minecraft.DenseBlockMetadata.newBuilder();
                                mBuilder.setIndex(lin);
                                for (Map.Entry<String, String> e : md.entrySet()) {
                                    mBuilder.putMetadata(e.getKey(), e.getValue());
                                }
                                metaList.add(mBuilder.build());
                            }
                        }
                    }
                }
            }

            Minecraft.DenseCube.Builder out = Minecraft.DenseCube.newBuilder();
            out.setSx(sx).setSy(sy).setSz(sz);
            out.setTypes(com.google.protobuf.ByteString.copyFrom(types));
            out.setBiomeIds(com.google.protobuf.ByteString.copyFrom(biomeIds));
            out.addAllMetadata(metaList);
            out.addAllBiomePalette(biomePalette);

            // Enable gzip compression for this large response if supported
            try {
                if (responseObserver instanceof io.grpc.stub.ServerCallStreamObserver) {
                    @SuppressWarnings("unchecked")
                    io.grpc.stub.ServerCallStreamObserver<Minecraft.DenseCube> scso = (io.grpc.stub.ServerCallStreamObserver<Minecraft.DenseCube>) responseObserver;
                    scso.setCompression("gzip");
                }
            } catch (Exception ignored) {}

            responseObserver.onNext(out.build());
            responseObserver.onCompleted();

            // Aggressively unload chunks we touched to avoid gradual memory/tick growth
            // try {
            //     int cminX = ((int)min.getX()) >> 4;
            //     int cmaxX = ((int)max.getX()) >> 4;
            //     int cminZ = ((int)min.getZ()) >> 4;
            //     int cmaxZ = ((int)max.getZ()) >> 4;
            //     for (int cx = cminX; cx <= cmaxX; cx++) {
            //         for (int cz = cminZ; cz <= cmaxZ; cz++) {
            //             java.util.Optional<org.spongepowered.api.world.Chunk> ch = world.getChunk(new Vector3i(cx, 0, cz));
            //             if (ch.isPresent()) {
            //                 world.unloadChunk(ch.get());
            //             }
            //         }
            //     }
            // } catch (Exception ignored) {}
        }).name("readDenseCubeWithMetadata").submit(plugin);
    }

    @Override
    public void readDenseCubeWithMajority(Cube cube, StreamObserver<Minecraft.DenseCubeMajority> responseObserver) {
        Task.builder().execute(() -> {
            World world = resolveWorld();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(cube.getMin(), cube.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();

            // Preload intersecting chunks before reading if requested by client
            try {
                if (cube.getPreloadChunks()) {
                    preloadChunksForWindow(world, min, max);
                }
            } catch (Exception ignored) {}

            int sx = (int)(max.getX() - min.getX() + 1);
            int sy = (int)(max.getY() - min.getY() + 1);
            int sz = (int)(max.getZ() - min.getZ() + 1);
            int total = sx * sy * sz;

            byte[] types = new byte[total];
            java.util.List<Minecraft.DenseBlockMetadata> metaList = new java.util.ArrayList<>();
            java.util.Map<String, Integer> biomeCounts = new java.util.HashMap<>();

            for (int xi = 0; xi < sx; xi++) {
                int x = (int)min.getX() + xi;
                for (int zi = 0; zi < sz; zi++) {
                    int z = (int)min.getZ() + zi;
                    String biomeFull = world.getBiome(x, (int)min.getY(), z).getId();
                    String biomeLabel = (biomeFull != null && biomeFull.contains(":")) ? biomeFull.split(":",2)[1] : (biomeFull != null ? biomeFull : "unknown");
                    biomeCounts.put(biomeLabel, biomeCounts.getOrDefault(biomeLabel, 0) + sy);
                    for (int yi = 0; yi < sy; yi++) {
                        int y = (int)min.getY() + yi;
                        int lin = ((xi * sy) + yi) * sz + zi;
                        String blockName = world.getBlock(x, y, z).getType().getName();
                        Byte idByte = blockNameToIdByte.get(blockName);
                        types[lin] = (idByte != null) ? idByte.byteValue() : (byte)0;

                        boolean relevant = false;
                        if (blockName != null) {
                            relevant = blockName.endsWith("_stairs") || blockName.contains("slab");
                        }
                        if (relevant) {
                            Location<World> location = world.getLocation(x, y, z);
                            Map<String, String> md = extractSlabAndStairMetadata(location);
                            if (md != null && !md.isEmpty()) {
                                Minecraft.DenseBlockMetadata.Builder mBuilder = Minecraft.DenseBlockMetadata.newBuilder();
                                mBuilder.setIndex(lin);
                                for (Map.Entry<String, String> e : md.entrySet()) {
                                    mBuilder.putMetadata(e.getKey(), e.getValue());
                                }
                                metaList.add(mBuilder.build());
                            }
                        }
                    }
                }
            }

            String majority = "unknown";
            int best = -1;
            for (Map.Entry<String, Integer> e : biomeCounts.entrySet()) {
                if (e.getValue() > best) {
                    best = e.getValue();
                    majority = e.getKey();
                }
            }

            Minecraft.DenseCubeMajority.Builder out = Minecraft.DenseCubeMajority.newBuilder();
            out.setSx(sx).setSy(sy).setSz(sz);
            out.setTypes(com.google.protobuf.ByteString.copyFrom(types));
            out.setMajorityBiome(majority);
            out.addAllMetadata(metaList);

            try {
                if (responseObserver instanceof io.grpc.stub.ServerCallStreamObserver) {
                    @SuppressWarnings("unchecked")
                    io.grpc.stub.ServerCallStreamObserver<Minecraft.DenseCubeMajority> scso = (io.grpc.stub.ServerCallStreamObserver<Minecraft.DenseCubeMajority>) responseObserver;
                    scso.setCompression("gzip");
                }
            } catch (Exception ignored) {}

            responseObserver.onNext(out.build());
            responseObserver.onCompleted();

            // Unload touched chunks
            // try {
            //     int cminX = ((int)min.getX()) >> 4;
            //     int cmaxX = ((int)max.getX()) >> 4;
            //     int cminZ = ((int)min.getZ()) >> 4;
            //     int cmaxZ = ((int)max.getZ()) >> 4;
            //     for (int cx = cminX; cx <= cmaxX; cx++) {
            //         for (int cz = cminZ; cz <= cmaxZ; cz++) {
            //             java.util.Optional<org.spongepowered.api.world.Chunk> ch = world.getChunk(new Vector3i(cx, 0, cz));
            //             if (ch.isPresent()) {
            //                 world.unloadChunk(ch.get());
            //             }
            //         }
            //     }
            // } catch (Exception ignored) {}
        }).name("readDenseCubeWithMajority").submit(plugin);
    }

    private Map<String, String> extractSlabAndStairMetadata(Location<World> loc) {
        Map<String, String> out = new HashMap<>();

        BlockState blockState = loc.getBlock();
        // Apply extended properties based on surroundings for accurate shapes/facing
        try {
            blockState = blockState.withExtendedProperties(loc);
        } catch (Exception ignored) {
        }

        // Slab: keys we observed via trait map ÿÿÿ half and variant
        String traitDump = stringifyTraitMap(blockState);
        if (!traitDump.isEmpty()) {
            for (String part : traitDump.split(",")) {
                if (part.startsWith("half=")) {
                    out.put("half", part.substring("half=".length()));
                } else if (part.startsWith("variant=")) {
                    out.put("variant", part.substring("variant=".length()));
                }
            }
        }

        // Stairs shape: prefer extended state but also compute a fallback/override from neighbors
        Optional<StairShape> stairShape = blockState.get(Keys.STAIR_SHAPE);
        String computedShape = null;
        Optional<Direction> facingOpt = blockState.get(Keys.DIRECTION);
        String halfStr = null;
        if (!traitDump.isEmpty()) {
            for (String part : traitDump.split(",")) {
                if (part.startsWith("half=")) {
                    halfStr = part.substring("half=".length());
                    break;
                }
            }
        }
        if (facingOpt.isPresent()) {
            Direction facingDir = facingOpt.get();
            Direction leftDir = turnLeft(facingDir);
            Direction rightDir = turnRight(facingDir);
            boolean leftPerp = isPerpendicularCorner(loc, leftDir, halfStr, leftDir);
            boolean rightPerp = isPerpendicularCorner(loc, rightDir, halfStr, rightDir);
            if (leftPerp ^ rightPerp) {
                computedShape = leftPerp ? "minecraft:outer_left" : "minecraft:outer_right";
            } else {
                computedShape = "minecraft:straight";
            }
        }
        if (stairShape.isPresent()) {
            out.put("shape", stairShape.get().getId());
        }
        if (computedShape != null) {
            out.put("shape", computedShape);
        }

        // Stairs facing direction: Keys.DIRECTION -> typically NORTH/EAST/SOUTH/WEST for stairs
        Optional<Direction> direction = blockState.get(Keys.DIRECTION);
        if (direction.isPresent()) {
            out.put("facing", direction.get().name());
        }

        // Stairs half (top/bottom): SpongeAPI 7.2 does not expose a dedicated StairHalf type/key.
        // Capture top/bottom via block trait map if present (e.g., half=top|bottom on stairs implementations)
        if (!traitDump.isEmpty()) {
            for (String part : traitDump.split(",")) {
                if (part.startsWith("half=")) {
                    out.put("half", part.substring("half=".length()));
                    break;
                }
            }
        }

        return out;
    }

    private boolean hasAdjacentStairs(Location<World> loc) {
        World world = loc.getExtent();
        int x = loc.getBlockX();
        int y = loc.getBlockY();
        int z = loc.getBlockZ();
        return isStairs(world.getLocation(x + 1, y, z).getBlock()) ||
               isStairs(world.getLocation(x - 1, y, z).getBlock()) ||
               isStairs(world.getLocation(x, y, z + 1).getBlock()) ||
               isStairs(world.getLocation(x, y, z - 1).getBlock());
    }

    private boolean isStairs(BlockState state) {
        String name = state.getType().getName();
        return name != null && name.endsWith("_stairs");
    }

    private Direction turnLeft(Direction dir) {
        switch (dir) {
            case NORTH: return Direction.WEST;
            case WEST: return Direction.SOUTH;
            case SOUTH: return Direction.EAST;
            case EAST: return Direction.NORTH;
            default: return dir;
        }
    }

    private Direction turnRight(Direction dir) {
        switch (dir) {
            case NORTH: return Direction.EAST;
            case EAST: return Direction.SOUTH;
            case SOUTH: return Direction.WEST;
            case WEST: return Direction.NORTH;
            default: return dir;
        }
    }

    private boolean isPerpendicularCorner(Location<World> base, Direction offsetDir, String halfWanted, Direction neighborFacingWanted) {
        Location<World> nloc = offset(base, offsetDir);
        BlockState nb = nloc.getBlock();
        if (!isStairs(nb)) return false;
        // match half via trait
        String nHalf = extractHalfTrait(nloc);
        if (halfWanted != null && nHalf != null && !halfWanted.equals(nHalf)) return false;
        Optional<Direction> nFacing = nb.get(Keys.DIRECTION);
        return nFacing.isPresent() && nFacing.get() == neighborFacingWanted;
    }

    private String extractHalfTrait(Location<World> loc) {
        try {
            BlockState st = loc.getBlock().withExtendedProperties(loc);
            String dump = stringifyTraitMap(st);
            for (String part : dump.split(",")) {
                if (part.startsWith("half=")) {
                    return part.substring("half=".length());
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    private Location<World> offset(Location<World> loc, Direction dir) {
        int x = loc.getBlockX();
        int y = loc.getBlockY();
        int z = loc.getBlockZ();
        switch (dir) {
            case NORTH: return loc.getExtent().getLocation(x, y, z - 1);
            case SOUTH: return loc.getExtent().getLocation(x, y, z + 1);
            case WEST: return loc.getExtent().getLocation(x - 1, y, z);
            case EAST: return loc.getExtent().getLocation(x + 1, y, z);
            default: return loc;
        }
    }

    private String stringifyTraitMap(BlockState blockState) {
        try {
            Map<BlockTrait<?>, ?> traitMap = blockState.getTraitMap();
            if (traitMap == null || traitMap.isEmpty()) {
                return "";
            }
            return traitMap.entrySet().stream()
                .map(e -> e.getKey().getName() + "=" + String.valueOf(e.getValue()))
                .sorted()
                .collect(Collectors.joining(","));
        } catch (Exception e) {
            return "";
        }
    }
    @Override
    public void initDataGen(Point point, StreamObserver<Empty> responseObserver) {
        Task.builder().execute(() -> {
            Player player = getConfiguredPlayer();
            player.offer(Keys.GAME_MODE, GameModes.SPECTATOR);
            World world = resolveWorld();
            // Set sunny weather.
            world.setWeather(Weathers.CLEAR, Long.MAX_VALUE);
            // Set daytime.
            world.getProperties().setWorldTime(1000);
            // Turn off daylight cycle.
            world.getProperties().setGameRule("DO_DAYLIGHT_CYCLE", "false");
            // Turn off weather cycle.
            world.getProperties().setGameRule("DO_WEATHER_CYCLE", "false");
            responseObserver.onNext(Empty.getDefaultInstance());
            responseObserver.onCompleted();
        }).name("initDataGen").submit(plugin);
    }

    @Override
    public void getHighestYAt(Point loc, StreamObserver<Point> responseObserver){
        Task.builder().execute(() -> {
            Point.Builder builder = Point.newBuilder();
            World world = resolveWorld();
            // Get tallest block at location
            int x = loc.getX();
            int y = world.getHighestYAt(loc.getX(), loc.getZ());
            int z = loc.getZ();
            builder.setX(x).setY(y).setZ(z);
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("getHighestAt").submit(plugin);
    }

    @Override
    public void getBiomeAt(Point loc, StreamObserver<BiomeResponse> responseObserver) {
        Task.builder().execute(() -> {
            BiomeResponse.Builder builder = BiomeResponse.newBuilder();
            World world = resolveWorld();
            
            // Get biome at location
            String biomeName = world.getBiome(loc.getX(), loc.getY(), loc.getZ()).getId();
            
            builder.setBiome(biomeName);
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("getBiomeAt").submit(plugin);
    }

    @Override
    public void setLoc(Point loc, StreamObserver<Point> responseObserver){
        Task.builder().execute(() -> {
            Point.Builder builder = Point.newBuilder();
            World world = resolveWorld();
            // Get tallest block at location
            int x = loc.getX();
            int y = loc.getY();
            int z = loc.getZ();
            builder.setX(x).setY(y).setZ(z);
            Player player = getConfiguredPlayer();
            Location<World> location = new Location<World>(world, x, y, z);
            // Boolean succ = player.setLocationAndRotation(location, rot_vec);
            Boolean succ = player.setLocation(location);
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("setLoc").submit(plugin);
    }

    @Override
    public void setPlayerLocRot(LocRotMsg locRotMsg, StreamObserver<Empty> responseObserver){
        Task.builder().execute(() -> {
            Player player = getConfiguredPlayer();
            World world = resolveWorld();
            Point loc = locRotMsg.getLocation();
            Point rot = locRotMsg.getRotation();
            Location<World> location = new Location<World>(world, loc.getX(), loc.getY(), loc.getZ());
            Vector3d rot_vec = new Vector3d(rot.getX(), rot.getY(), rot.getZ());
            // player.setRotation(rot_vec);
            // Boolean succ = player.setLocation(location);
            Boolean succ = player.setLocationAndRotation(location, rot_vec);
            responseObserver.onNext(Empty.getDefaultInstance());
            responseObserver.onCompleted();
        }).name("setLocRot").submit(plugin);
    }

    @Override
    public void setLocY(Point loc, StreamObserver<Point> responseObserver){
        Task.builder().execute(() -> {
            Point.Builder builder = Point.newBuilder();
            World world = resolveWorld();
            // Get tallest block at location
            int x = loc.getX();
            int y = loc.getY();
            int z = loc.getZ();
            builder.setX(x).setY(y).setZ(z);
            Player player = getConfiguredPlayer();
            Location<World> location = new Location<World>(world, x, y, z);
            // Boolean succ = player.setLocationAndRotation(location, rot_vec);
            Boolean succ = player.setLocation(location);
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }).name("setLocY").submit(plugin);
    }

    @Override
    public void setRot(Point rot, StreamObserver<Empty> responseObserver){
        Task.builder().execute(() -> {
            Player player = getConfiguredPlayer();
            Vector3d rot_vec = new Vector3d(rot.getX(), rot.getY(), rot.getZ());
            player.setRotation(rot_vec);
            responseObserver.onNext(Empty.getDefaultInstance());
            responseObserver.onCompleted();
        }).name("setRot").submit(plugin);
    }

    @Override
    public void fillCube(FillCubeRequest request, StreamObserver<Empty> responseObserver) {
        Task.builder().execute(() -> {
            World world = resolveWorld();
            Cube c = request.getCube();
            Pair<Vector3d, Vector3d> boundaries = findCubeBoundaries(c.getMin(), c.getMax());
            Vector3d min = boundaries.getLeft();
            Vector3d max = boundaries.getRight();
            BlockType type;
            try {
                Field typeField = BlockTypes.class.getField(request.getType().toString());
                type = (BlockType) typeField.get(null);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }

            for (int x = (int)min.getX(); x <= max.getX(); x++) {
                for (int y = (int)min.getY(); y <= max.getY(); y++) {
                    for (int z = (int)min.getZ(); z <= max.getZ(); z++) {
                        world.setBlockType(x, y, z, type);
                    }
                }
            }
            responseObserver.onNext(Empty.getDefaultInstance());
            responseObserver.onCompleted();
        }).name("fillCube").submit(plugin);
    }

    private Pair<Vector3d, Vector3d> findCubeBoundaries(Point p1, Point p2){
        int minX = Math.min(p1.getX(), p2.getX());
        int minY = Math.min(p1.getY(), p2.getY());
        int minZ = Math.min(p1.getZ(), p2.getZ());
        Vector3d min = new Vector3d(minX, minY, minZ);
        int maxX = Math.max(p1.getX(), p2.getX());
        int maxY = Math.max(p1.getY(), p2.getY());
        int maxZ = Math.max(p1.getZ(), p2.getZ());
        Vector3d max = new Vector3d(maxX, maxY, maxZ);

        return new ImmutablePair<>(min, max);
    }

    public void setOrientation(Location<World> blockLoc, Orientation orientation, BlockType btype) throws IllegalStateException{
        Optional<DirectionalData> optionalData = blockLoc.get(DirectionalData.class);
        if (!optionalData.isPresent()) {
            throw new IllegalStateException("Failed to get block location data");
        }
        DirectionalData data = optionalData.get();
        data.set(Keys.DIRECTION, Direction.valueOf(orientation.toString()));
        BlockState state = btype.getDefaultState();
        Optional<BlockState> newState = state.with(data.asImmutable());
        if (!newState.isPresent()) {
            throw new IllegalStateException("block type " + btype.toString() + " failed to set orientation!");
        }
        blockLoc.setBlock(newState.get());
    }

    private void registerCommands() {
        CommandSpec setAreaCommand = CommandSpec.builder()
            .description(Text.of("Set up the building areas for the AI model"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    Location<World> loc = player.getLocation();
                    
                    setModelArea(SetAreaRequest.newBuilder()
                        .setOrigin(Point.newBuilder()
                            .setX((int)loc.getX())
                            .setY((int)loc.getY())
                            .setZ((int)loc.getZ()))
                        .setUseParticles(true)
                        .build(),
                        new StreamObserver<TriggerResponse>() {
                            @Override
                            public void onNext(TriggerResponse response) {
                                if (response.getSuccess()) {
                                    player.sendMessage(Text.of("Building areas set up!"));
                                }
                            }
                            
                            @Override
                            public void onError(Throwable t) {
                                player.sendMessage(Text.of("Error setting up areas: " + t.getMessage()));
                            }
                            
                            @Override
                            public void onCompleted() {}
                        });
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec readAreaCommand = CommandSpec.builder()
            .description(Text.of("Read the current state of both building areas"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    
                    if (currentBuildingArea == null) {
                        player.sendMessage(Text.of("Building areas not set! Use /set-area first"));
                        return CommandResult.success();
                    }

                    // Read first latent area
                    Point firstOrigin = currentBuildingArea.getFirstLatentOrigin();
                    Cube firstCube = Cube.newBuilder()
                        .setMin(firstOrigin)
                        .setMax(Point.newBuilder()
                            .setX(firstOrigin.getX() + LATENT_SIZE - 1)
                            .setY(firstOrigin.getY() + LATENT_SIZE - 1)
                            .setZ(firstOrigin.getZ() + LATENT_SIZE - 1))
                        .build();

                    readCube(firstCube, new StreamObserver<Blocks>() {
                        @Override
                        public void onNext(Blocks blocks) {
                            player.sendMessage(Text.of("First area blocks by layer:"));
                            printCubeLayers(player, blocks);
                        }
                        @Override
                        public void onError(Throwable t) {
                            player.sendMessage(Text.of("Error reading first area: " + t.getMessage()));
                        }
                        @Override
                        public void onCompleted() {}
                    });

                    // Read second latent area
                    Point secondOrigin = currentBuildingArea.getSecondLatentOrigin();
                    Cube secondCube = Cube.newBuilder()
                        .setMin(secondOrigin)
                        .setMax(Point.newBuilder()
                            .setX(secondOrigin.getX() + LATENT_SIZE - 1)
                            .setY(secondOrigin.getY() + LATENT_SIZE - 1)
                            .setZ(secondOrigin.getZ() + LATENT_SIZE - 1))
                        .build();

                    readCube(secondCube, new StreamObserver<Blocks>() {
                        @Override
                        public void onNext(Blocks blocks) {
                            player.sendMessage(Text.of("Second area blocks by layer:"));
                            printCubeLayers(player, blocks);
                        }
                        @Override
                        public void onError(Throwable t) {
                            player.sendMessage(Text.of("Error reading second area: " + t.getMessage()));
                        }
                        @Override
                        public void onCompleted() {}
                    });
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec triggerModelCommand = CommandSpec.builder()
            .description(Text.of("Trigger the AI model to process your build"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    triggerModel(Empty.getDefaultInstance(), 
                        new StreamObserver<TriggerResponse>() {
                            @Override
                            public void onNext(TriggerResponse response) {
                                if (response.getSuccess()) {
                                    player.sendMessage(Text.of("Model processing complete!"));
                                } else {
                                    player.sendMessage(Text.of("Error: " + response.getError()));
                                }
                            }
                            @Override
                            public void onError(Throwable t) {
                                player.sendMessage(Text.of("Error triggering model: " + t.getMessage()));
                            }
                            @Override
                            public void onCompleted() {}
                        });
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec loadPresetCommand = CommandSpec.builder()
            .description(Text.of("Load a preset terrain configuration"))
            .arguments(GenericArguments.string(Text.of("name")))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    String presetName = args.<String>getOne("name").get();
                    
                    // Create the load preset request
                    LoadPresetRequest request = LoadPresetRequest.newBuilder()
                        .setName(presetName)
                        .build();
                    
                    // Call the model service
                    ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 5002)
                        .usePlaintext()
                        .build();
                    
                    MinecraftServiceGrpc.MinecraftServiceStub modelStub = 
                        MinecraftServiceGrpc.newStub(channel);
                    
                    modelStub.handleLoadPreset(request, new StreamObserver<TriggerResponse>() {
                        @Override
                        public void onNext(TriggerResponse response) {
                            if (response.getSuccess()) {
                                player.sendMessage(Text.of("Preset loaded successfully!"));
                            } else {
                                player.sendMessage(Text.of("Error: " + response.getError()));
                            }
                        }
                        
                        @Override
                        public void onError(Throwable t) {
                            player.sendMessage(Text.of("Error loading preset: " + t.getMessage()));
                            channel.shutdown();
                        }
                        
                        @Override
                        public void onCompleted() {
                            channel.shutdown();
                        }
                    });
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec raiseToolCommand = CommandSpec.builder()
            .description(Text.of("Activate the terrain raise tool"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    if (currentBuildingArea == null) {
                        player.sendMessage(Text.of("Building areas not set! Use /set-area first"));
                        return CommandResult.success();
                    }
                    terrainTool.activate(TerrainTool.ToolType.RAISE);
                    player.sendMessage(Text.of("Terrain raise tool activated! Right-click to select areas."));
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec lowerToolCommand = CommandSpec.builder()
            .description(Text.of("Activate the terrain lower tool"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    if (currentBuildingArea == null) {
                        player.sendMessage(Text.of("Building areas not set! Use /set-area first"));
                        return CommandResult.success();
                    }
                    terrainTool.activate(TerrainTool.ToolType.LOWER);
                    player.sendMessage(Text.of("Terrain lower tool activated! Right-click to select areas."));
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec cancelToolCommand = CommandSpec.builder()
            .description(Text.of("Cancel the current terrain tool"))
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    if (!terrainTool.isActive()) {
                        player.sendMessage(Text.of("No terrain tool is currently active!"));
                        return CommandResult.success();
                    }
                    terrainTool.deactivate();
                    player.sendMessage(Text.of("Terrain tool deactivated. All selections cleared."));
                }
                return CommandResult.success();
            })
            .build();

        CommandSpec triggerToolCommand = CommandSpec.builder()
            .description(Text.of("Trigger the terrain modification tool"))
            .arguments(
                GenericArguments.optional(
                    GenericArguments.integer(Text.of("axis")), 1),
                GenericArguments.optional(
                    GenericArguments.integer(Text.of("amount")), 1),
                GenericArguments.optional(
                    GenericArguments.bool(Text.of("shift_style")), false)
            )
            .executor((src, args) -> {
                if (src instanceof Player) {
                    Player player = (Player) src;
                    
                    if (!terrainTool.isActive()) {
                        player.sendMessage(Text.of("No terrain tool is currently active! Use /raise-tool or /lower-tool first."));
                        return CommandResult.success();
                    }

                    if (!terrainTool.hasSelections()) {
                        player.sendMessage(Text.of("?cNo areas selected! ?rRight-click blocks in the output area to select 4x4x4 regions to modify."));
                        return CommandResult.success();
                    }

                    // Get command parameters
                    int shift_axis = args.<Integer>getOne("axis").orElse(1);
                    int shift_amount = args.<Integer>getOne("amount").orElse(1);
                    boolean shift_style = args.<Boolean>getOne("shift_style").orElse(false);

                    // Get the tool type and selected indices
                    TerrainTool.ToolType toolType = terrainTool.getCurrentToolType();
                    String action = (toolType == TerrainTool.ToolType.RAISE) ? "Raising" : "Lowering";
                    String indices = terrainTool.getSelectedIndicesString();
                    
                    // Inform player of the action
                    player.sendMessage(Text.of(String.format(
                        "?a%s terrain at indices: ?6%s ?r(axis=%d, amount=%d, shift_style=%b)", 
                        action, 
                        indices,
                        shift_axis,
                        shift_amount,
                        shift_style
                    )));
                    
                    // Create channel to model service
                    ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 5002)
                        .usePlaintext()
                        .build();
                    
                    MinecraftServiceGrpc.MinecraftServiceStub modelStub = 
                        MinecraftServiceGrpc.newStub(channel);
                    
                    // Convert selected areas to modification request
                    TerrainModificationRequest.Builder requestBuilder = TerrainModificationRequest.newBuilder();
                    
                    // Add each selected region to the request
                    for (Vector3i region : terrainTool.getSelectedAreas()) {
                        RegionModification.Builder regionBuilder = RegionModification.newBuilder()
                            .setOrigin(Point.newBuilder()
                                .setX((region.getX() - currentBuildingArea.getOutputOrigin().getX()) / 4)
                                .setY((region.getY() - currentBuildingArea.getOutputOrigin().getY()) / 4)
                                .setZ((region.getZ() - currentBuildingArea.getOutputOrigin().getZ()) / 4))
                            .setType(toolType == TerrainTool.ToolType.RAISE ? 
                                ModificationType.RAISE : ModificationType.LOWER)
                            .setShiftAmount(shift_amount)
                            .setShiftStyle(shift_style);
                            
                        requestBuilder.addRegions(regionBuilder);
                    }
                    
                    // Add shift axis to request
                    requestBuilder.setShiftAxis(shift_axis);
                        
                    // Send request to model service
                    modelStub.modifyTerrain(requestBuilder.build(), new StreamObserver<TriggerResponse>() {
                        @Override
                        public void onNext(TriggerResponse response) {
                            if (response.getSuccess()) {
                                player.sendMessage(Text.of("?aTerrain modification complete!"));
                            } else {
                                player.sendMessage(Text.of("?cError: " + response.getError()));
                            }
                        }
                        
                        @Override
                        public void onError(Throwable t) {
                            player.sendMessage(Text.of("?cError modifying terrain: " + t.getMessage()));
                            channel.shutdown();
                        }
                        
                        @Override
                        public void onCompleted() {
                            channel.shutdown();
                            // Deactivate tool after successful modification
                            terrainTool.deactivate();
                            player.sendMessage(Text.of("?7Tool deactivated. Use /raise-tool or /lower-tool to start a new selection."));
                        }
                    });
                }
                return CommandResult.success();
            })
            .build();

        // Inspect the traits of a simple target: block under the player's feet (fallbacks if air)
        CommandSpec inspectTraitsCommand = CommandSpec.builder()
            .description(Text.of("Inspect BlockState trait map for the block under you"))
            .executor((src, args) -> {
                if (!(src instanceof Player)) {
                    src.sendMessage(Text.of("Player only command"));
                    return CommandResult.success();
                }
                Player player = (Player) src;
                Location<World> playerLoc = player.getLocation();
                Location<World> targetLoc = new Location<>(playerLoc.getExtent(), playerLoc.getBlockX(), playerLoc.getBlockY() - 1, playerLoc.getBlockZ());
                BlockState state = targetLoc.getBlock();

                if (state.getType() == BlockTypes.AIR) {
                    targetLoc = new Location<>(playerLoc.getExtent(), playerLoc.getBlockX(), playerLoc.getBlockY(), playerLoc.getBlockZ());
                    state = targetLoc.getBlock();
                }
                if (state.getType() == BlockTypes.AIR) {
                    targetLoc = new Location<>(playerLoc.getExtent(), playerLoc.getBlockX(), playerLoc.getBlockY() - 2, playerLoc.getBlockZ());
                    state = targetLoc.getBlock();
                }

                BlockState extended = state;
                try {
                    extended = state.withExtendedProperties(targetLoc);
                } catch (Exception ignored) {
                }
                String dump = stringifyTraitMap(extended);
                player.sendMessage(Text.of("Block: " + extended.getType().getName() +
                    " at (" + targetLoc.getBlockX() + "," + targetLoc.getBlockY() + "," + targetLoc.getBlockZ() + ")"));
                player.sendMessage(Text.of(dump.isEmpty() ? "Traits (extended): <none>" : ("Traits (extended): " + dump)));
                Optional<StairShape> shape = extended.get(Keys.STAIR_SHAPE);
                if (shape.isPresent()) {
                    player.sendMessage(Text.of("Stair shape (extended): " + shape.get().getId()));
                }
                return CommandResult.success();
            })
            .build();

        Sponge.getCommandManager().register(plugin, setAreaCommand, "set-area", "sa");
        Sponge.getCommandManager().register(plugin, readAreaCommand, "read-area", "ra");
        Sponge.getCommandManager().register(plugin, triggerModelCommand, "trigger-model", "tm");
        Sponge.getCommandManager().register(plugin, loadPresetCommand, "load-preset", "lp");
        Sponge.getCommandManager().register(plugin, raiseToolCommand, "raise-tool", "rt");
        Sponge.getCommandManager().register(plugin, lowerToolCommand, "lower-tool", "lt");
        Sponge.getCommandManager().register(plugin, cancelToolCommand, "cancel-tool", "ct");
        Sponge.getCommandManager().register(plugin, triggerToolCommand, "trigger-tool", "tt");
        Sponge.getCommandManager().register(plugin, inspectTraitsCommand, "inspect-traits", "it");

        // /terraform <biome>
        CommandSpec terraformCommand = CommandSpec.builder()
            .description(Text.of("Generate a 16x16x16 terrain cube 9 blocks ahead, centered on terrain"))
            .arguments(GenericArguments.string(Text.of("biome")))
            .executor((src, args) -> {
                if (!(src instanceof Player)) {
                    src.sendMessage(Text.of("Player only command"));
                    return CommandResult.success();
                }

                Player player = (Player) src;
                World world = resolveWorld();

                // Determine facing cardinal direction from yaw
                double yaw = player.getRotation().getY();
                // Normalize to (-180, 180]
                while (yaw <= -180) yaw += 360;
                while (yaw > 180) yaw -= 360;

                Direction facing;
                if (yaw > -45 && yaw <= 45) {
                    facing = Direction.SOUTH; // +Z
                } else if (yaw > 45 && yaw <= 135) {
                    facing = Direction.WEST;  // -X
                } else if (yaw <= -135 || yaw > 135) {
                    facing = Direction.NORTH; // -Z
                } else {
                    facing = Direction.EAST;  // +X
                }

                // Player block position
                int px = player.getLocation().getBlockX();
                int py = player.getLocation().getBlockY();
                int pz = player.getLocation().getBlockZ();

                // Compute center 9 blocks ahead (8 to center 16, +1 buffer)
                int cx = px;
                int cz = pz;
                switch (facing) {
                    case NORTH:
                        cz = pz - 9;
                        break;
                    case SOUTH:
                        cz = pz + 9;
                        break;
                    case WEST:
                        cx = px - 9;
                        break;
                    case EAST:
                        cx = px + 9;
                        break;
                    default:
                        break;
                }

                // Highest Y at the center XZ
                int highestY = world.getHighestYAt(cx, cz);

                // Build origin for 16x16x16 cube so that (cx, highestY, cz) is the exact vertical center
                final int size = 16;
                int ox = cx - size / 2;
                int oy = highestY - size / 2;
                int oz = cz - size / 2;
                if (oy < 1) oy = 1; // keep above bedrock/void

                String biomeArg = args.<String>getOne("biome").get();

                // Prepare request to Python model server
                ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 5002)
                    .usePlaintext()
                    .build();

                MinecraftServiceGrpc.MinecraftServiceStub modelStub = MinecraftServiceGrpc.newStub(channel);

                // Start outline while generation runs
                Point terraformOrigin = Minecraft.Point.newBuilder().setX(ox).setY(oy).setZ(oz).build();
                startTerraformOutline(terraformOrigin, size);

                Minecraft.TerraformRequest.Builder req = Minecraft.TerraformRequest.newBuilder()
                    .setOrigin(terraformOrigin)
                    .setSize(size);

                // If biome is an int, treat as id; otherwise a label
                try {
                    int biomeId = Integer.parseInt(biomeArg);
                    req.setBiomeId(biomeId);
                } catch (NumberFormatException nfe) {
                    req.setBiomeLabel(biomeArg);
                }

                player.sendMessage(Text.of(String.format(
                    "Terraforming biome '%s' at origin (%d,%d,%d) (center %d blocks ahead, highestY=%d)",
                    biomeArg, ox, oy, oz, 9, highestY)));

                modelStub.terraform(req.build(), new StreamObserver<Minecraft.TriggerResponse>() {
                    @Override
                    public void onNext(Minecraft.TriggerResponse response) {
                        stopTerraformOutline();
                        if (response.getSuccess()) {
                            player.sendMessage(Text.of("?aTerraform complete."));
                        } else {
                            player.sendMessage(Text.of("?cTerraform error: " + response.getError()));
                        }
                    }

                    @Override
                    public void onError(Throwable t) {
                        stopTerraformOutline();
                        player.sendMessage(Text.of("?cError contacting model server: " + t.getMessage()));
                        channel.shutdown();
                    }

                    @Override
                    public void onCompleted() {
                        stopTerraformOutline();
                        channel.shutdown();
                    }
                });

                return CommandResult.success();
            })
            .build();

        Sponge.getCommandManager().register(plugin, terraformCommand, "terraform");
    }

    private void printCubeLayers(Player player, Blocks blocks) {
        // Create a 3D array to store the blocks
        Minecraft.BlockType[][][] cube = new Minecraft.BlockType[LATENT_SIZE][LATENT_SIZE][LATENT_SIZE];
        
        // Fill the 3D array with AIR by default
        for (int y = 0; y < LATENT_SIZE; y++) {
            for (int x = 0; x < LATENT_SIZE; x++) {
                for (int z = 0; z < LATENT_SIZE; z++) {
                    cube[y][x][z] = Minecraft.BlockType.AIR;
                }
            }
        }
        
        // Get the minimum coordinates to calculate relative positions
        Point min = blocks.getBlocks(0).getPosition();
        int minX = min.getX();
        int minY = min.getY();
        int minZ = min.getZ();
        
        // Fill in the actual blocks
        for (Block block : blocks.getBlocksList()) {
            Point pos = block.getPosition();
            // Convert to relative coordinates
            int relX = pos.getX() - minX;
            int relY = pos.getY() - minY;
            int relZ = pos.getZ() - minZ;
            
            // Verify the coordinates are in range
            if (relX >= 0 && relX < LATENT_SIZE && 
                relY >= 0 && relY < LATENT_SIZE && 
                relZ >= 0 && relZ < LATENT_SIZE) {
                cube[relY][relX][relZ] = block.getType();
            }
        }
        
        // Print each layer
        for (int y = 0; y < LATENT_SIZE; y++) {
            player.sendMessage(Text.of(String.format("Layer %d (y=%d):", y, y)));
            
            // Print each row of this layer
            for (int x = 0; x < LATENT_SIZE; x++) {
                StringBuilder rowMsg = new StringBuilder();
                for (int z = 0; z < LATENT_SIZE; z++) {
                    Minecraft.BlockType type = cube[y][x][z];
                    rowMsg.append(String.format("%-8s", type));  // Left align with 8 chars width
                }
                player.sendMessage(Text.of(rowMsg.toString()));
            }
            player.sendMessage(Text.of("")); // Empty line between layers
        }
    }

    private void clearPreviousAreas() {
        // Cancel any running particle tasks
        if (particleTask != null) {
            particleTask.cancel();
            particleTask = null;
        }

        // If we have previous area coordinates, clear those blocks
        if (currentBuildingArea != null) {
            World world = resolveWorld();
            
            // Clear first latent area
            clearArea(world, currentBuildingArea.getFirstLatentOrigin(), LATENT_SIZE + 2);
            
            // Clear second latent area
            clearArea(world, currentBuildingArea.getSecondLatentOrigin(), LATENT_SIZE + 2);
            
            // Clear output area
            clearArea(world, currentBuildingArea.getOutputOrigin(), OUTPUT_SIZE + 2);
            
            currentBuildingArea = null;
        }
    }

    private void clearArea(World world, Point origin, int size) {
        // Clear a cubic area (including the glass shell if it exists)
        for (int x = -1; x <= size; x++) {
            for (int y = -1; y <= size; y++) {
                for (int z = -1; z <= size; z++) {
                    world.getLocation(
                        origin.getX() + x,
                        origin.getY() + y,
                        origin.getZ() + z
                    ).setBlockType(BlockTypes.AIR);
                }
            }
        }
    }

    @Override
    public void getCurrentBuildingArea(Empty request, StreamObserver<BuildingArea> responseObserver) {
        Task.builder().execute(() -> {
            if (currentBuildingArea != null) {
                responseObserver.onNext(currentBuildingArea);
            } else {
                responseObserver.onNext(BuildingArea.getDefaultInstance());
            }
            responseObserver.onCompleted();
        }).name("getCurrentBuildingArea").submit(plugin);
    }

    @Override
    public void triggerModel(Empty request, StreamObserver<TriggerResponse> responseObserver) {
        Task.builder().execute(() -> {
            try {
                // Create a channel to our Python model server
                ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 5002)
                    .usePlaintext()
                    .build();
                
                // Create a stub for the model service
                MinecraftServiceGrpc.MinecraftServiceStub modelStub = MinecraftServiceGrpc.newStub(channel);
                
                // Call the model service
                modelStub.triggerModel(Empty.getDefaultInstance(), new StreamObserver<TriggerResponse>() {
                    @Override
                    public void onNext(TriggerResponse response) {
                        responseObserver.onNext(response);
                    }
                    
                    @Override
                    public void onError(Throwable t) {
                        logger.error("Error calling model service: " + t.getMessage());
                        responseObserver.onNext(TriggerResponse.newBuilder()
                            .setSuccess(false)
                            .setError("Failed to call model service: " + t.getMessage())
                            .build());
                        responseObserver.onCompleted();
                    }
                    
                    @Override
                    public void onCompleted() {
                        responseObserver.onCompleted();
                        channel.shutdown();
                    }
                });
            } catch (Exception e) {
                logger.error("Error in triggerModel: " + e.getMessage());
                responseObserver.onNext(TriggerResponse.newBuilder()
                    .setSuccess(false)
                    .setError("Internal error: " + e.getMessage())
                    .build());
                responseObserver.onCompleted();
            }
        }).name("triggerModel").submit(plugin);
    }

    @Listener
    public void onInteractBlockSecondary(InteractBlockEvent.Secondary event) {
        logger.info("Block interaction detected");
        
        if (!terrainTool.isActive() || !(event.getCause().root() instanceof Player)) {
            logger.info("Tool not active or not a player");
            return;
        }

        Player player = (Player) event.getCause().root();
        Location<World> clickedLoc = event.getTargetBlock().getLocation().get();

        // Check if click is within the output area
        if (!isWithinOutputArea(clickedLoc)) {
            logger.info("Click outside output area");
            player.sendMessage(Text.of("Click must be within the output area!"));
            return;
        }

        // Calculate which 4x4x4 region was clicked
        Vector3i regionOrigin = calculateRegionOrigin(clickedLoc);
        
        // Add region to selected areas if not already selected
        if (terrainTool.addSelectedArea(regionOrigin)) {
            player.sendMessage(Text.of("Selected region at " + regionOrigin));
            showRegionHighlight(regionOrigin);
        }

        // Cancel the original interaction
        event.setCancelled(true);
    }

    private boolean isWithinOutputArea(Location<World> loc) {
        if (currentBuildingArea == null) {
            logger.info("Building area is null");
            return false;
        }
        
        Point outputOrigin = currentBuildingArea.getOutputOrigin();
        int x = loc.getBlockX();
        int y = loc.getBlockY();
        int z = loc.getBlockZ();
        
        // Add debug logging
        logger.info("Click location: " + x + ", " + y + ", " + z);
        logger.info("Output origin: " + outputOrigin.getX() + ", " + outputOrigin.getY() + ", " + outputOrigin.getZ());
        logger.info("Output size: " + OUTPUT_SIZE);
        
        boolean inX = x >= outputOrigin.getX() && x < outputOrigin.getX() + OUTPUT_SIZE;
        boolean inY = y >= outputOrigin.getY() && y < outputOrigin.getY() + OUTPUT_SIZE;
        boolean inZ = z >= outputOrigin.getZ() && z < outputOrigin.getZ() + OUTPUT_SIZE;
        
        // Add more detailed debug info
        logger.info("In X range: " + inX + " (" + outputOrigin.getX() + " <= " + x + " < " + (outputOrigin.getX() + OUTPUT_SIZE) + ")");
        logger.info("In Y range: " + inY + " (" + outputOrigin.getY() + " <= " + y + " < " + (outputOrigin.getY() + OUTPUT_SIZE) + ")");
        logger.info("In Z range: " + inZ + " (" + outputOrigin.getZ() + " <= " + z + " < " + (outputOrigin.getZ() + OUTPUT_SIZE) + ")");
        
        return inX && inY && inZ;
    }

    private Vector3i calculateRegionOrigin(Location<World> loc) {
        Point outputOrigin = currentBuildingArea.getOutputOrigin();
        
        // Calculate relative position within output area
        int relX = loc.getBlockX() - outputOrigin.getX();
        int relY = loc.getBlockY() - outputOrigin.getY();
        int relZ = loc.getBlockZ() - outputOrigin.getZ();
        
        // Calculate region indices (integer division by 4 to get 4x4x4 regions)
        int regionX = (relX / 4) * 4 + outputOrigin.getX();
        int regionY = (relY / 4) * 4 + outputOrigin.getY();
        int regionZ = (relZ / 4) * 4 + outputOrigin.getZ();
        
        return new Vector3i(regionX, regionY, regionZ);
    }

    private void showRegionHighlight(Vector3i origin) {
        World world = resolveWorld();
        
        // Show particle effects around the 4x4x4 region
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                for (int z = 0; z < 4; z++) {
                    // Only show particles on the edges of the region
                    if (x == 0 || x == 3 || y == 0 || y == 3 || z == 0 || z == 3) {
                        world.spawnParticles(
                            ParticleEffect.builder()
                                .type(ParticleTypes.REDSTONE_DUST)
                                .build(),
                            new Vector3d(
                                origin.getX() + x + 0.5,
                                origin.getY() + y + 0.5,
                                origin.getZ() + z + 0.5
                            )
                        );
                    }
                }
            }
        }
    }

}

// public class TerrainTool {
//     private boolean isActive = false;
//     private ToolType currentToolType = null;
//     private Set<Vector3i> selectedAreas = new HashSet<>(); // Stores the origin points of 4x4x4 areas
//     private Task particleTask = null;
    
//     public enum ToolType {
//         RAISE,
//         LOWER
//     }

//     // Methods for managing tool state
//     public void activate(ToolType type) {
//         this.isActive = true;
//         this.currentToolType = type;
//         this.selectedAreas.clear();
//     }

//     public void deactivate() {
//         this.isActive = false;
//         this.currentToolType = null;
//         this.selectedAreas.clear();
//         if (particleTask != null) {
//             particleTask.cancel();
//             particleTask = null;
//         }
//     }

//     public boolean isActive() {
//         return isActive;
//     }

//     public ToolType getCurrentToolType() {
//         return currentToolType;
//     }
// }
