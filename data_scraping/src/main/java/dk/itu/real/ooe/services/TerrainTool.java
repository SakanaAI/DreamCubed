package dk.itu.real.ooe.services;

import org.spongepowered.api.scheduler.Task;
import org.spongepowered.api.plugin.PluginContainer;
import org.spongepowered.api.Sponge;
import org.spongepowered.api.effect.particle.ParticleEffect;
import org.spongepowered.api.effect.particle.ParticleTypes;
import org.spongepowered.api.world.World;
import com.flowpowered.math.vector.Vector3d;
import com.flowpowered.math.vector.Vector3i;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Remove any imports referencing dk.itu.real.ooe.models
// Add these correct imports instead:
import dk.itu.real.ooe.Minecraft;
import dk.itu.real.ooe.Minecraft.BuildingArea;
import dk.itu.real.ooe.Minecraft.Point;

public class TerrainTool {
    // Add logger as class field
    private static final Logger logger = LoggerFactory.getLogger(TerrainTool.class);
    
    private boolean isActive = false;
    private ToolType currentToolType = null;
    private Set<Vector3i> selectedAreas = new HashSet<>(); // Stores the origin points of 4x4x4 areas
    private Task particleTask = null;
    private final PluginContainer plugin;
    private BuildingArea buildingArea; // Store reference to current building area
    
    public enum ToolType {
        RAISE,
        LOWER
    }

    public TerrainTool(PluginContainer plugin) {
        this.plugin = plugin;
    }

    // Methods for managing tool state
    public void activate(ToolType type) {
        this.isActive = true;
        this.currentToolType = type;
        this.selectedAreas.clear();
        startParticleTask();
    }

    public void deactivate() {
        this.isActive = false;
        this.currentToolType = null;
        this.selectedAreas.clear();
        stopParticleTask();
    }

    public boolean isActive() {
        return isActive;
    }

    public ToolType getCurrentToolType() {
        return currentToolType;
    }

    // Add this new method
    public boolean addSelectedArea(Vector3i region) {
        boolean added = selectedAreas.add(region);
        if (added) {
            // Log indices only when a new area is selected
            Point outputOrigin = buildingArea.getOutputOrigin();
            int relX = (region.getX() - outputOrigin.getX()) / 4;
            int relY = (region.getY() - outputOrigin.getY()) / 4;
            int relZ = (region.getZ() - outputOrigin.getZ()) / 4;
            
            logger.info(String.format(
                "Selected region corresponds to latent indices: [%d, %d, %d]", 
                relX, relY, relZ
            ));
        }
        return added;
    }

    // Add getter for selected areas if needed
    public Set<Vector3i> getSelectedAreas() {
        return new HashSet<>(selectedAreas);  // Return a copy to prevent external modification
    }

    // Add method to update building area
    public void setBuildingArea(BuildingArea area) {
        this.buildingArea = area;
    }

    // Add method to show latent correspondence
    private void showLatentCorrespondence(World world, Vector3i outputRegion) {
        Point outputOrigin = buildingArea.getOutputOrigin();
        int relX = (outputRegion.getX() - outputOrigin.getX()) / 4;
        int relY = (outputRegion.getY() - outputOrigin.getY()) / 4;
        int relZ = (outputRegion.getZ() - outputOrigin.getZ()) / 4;

        // Show particles at corresponding positions in both latent spaces
        showLatentPosition(world, buildingArea.getFirstLatentOrigin(), relX, relY, relZ);
        showLatentPosition(world, buildingArea.getSecondLatentOrigin(), relX, relY, relZ);
    }

    private void showLatentPosition(World world, Point latentOrigin, int relX, int relY, int relZ) {
        // Calculate absolute position in latent space
        double x = latentOrigin.getX() + relX;
        double y = latentOrigin.getY() + relY;
        double z = latentOrigin.getZ() + relZ;
        
        ParticleEffect effect = ParticleEffect.builder()
            .type(ParticleTypes.DRAGON_BREATH)
            .quantity(1)
            .build();
        
        // Spawn particles at the 8 corners of the block
        double[][] corners = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
        };
        
        for (double[] corner : corners) {
            world.spawnParticles(
                effect,
                new Vector3d(
                    x + corner[0],
                    y + corner[1],
                    z + corner[2]
                )
            );
        }

        // Also spawn particles along the edges for better visibility
        for (double t = 0.2; t <= 0.8; t += 0.2) {
            // Vertical edges
            world.spawnParticles(effect, new Vector3d(x, y + t, z));
            world.spawnParticles(effect, new Vector3d(x + 1, y + t, z));
            world.spawnParticles(effect, new Vector3d(x, y + t, z + 1));
            world.spawnParticles(effect, new Vector3d(x + 1, y + t, z + 1));
            
            // Horizontal edges (X)
            world.spawnParticles(effect, new Vector3d(x + t, y, z));
            world.spawnParticles(effect, new Vector3d(x + t, y + 1, z));
            world.spawnParticles(effect, new Vector3d(x + t, y, z + 1));
            world.spawnParticles(effect, new Vector3d(x + t, y + 1, z + 1));
            
            // Horizontal edges (Z)
            world.spawnParticles(effect, new Vector3d(x, y, z + t));
            world.spawnParticles(effect, new Vector3d(x + 1, y, z + t));
            world.spawnParticles(effect, new Vector3d(x, y + 1, z + t));
            world.spawnParticles(effect, new Vector3d(x + 1, y + 1, z + t));
        }
    }

    private void startParticleTask() {
        stopParticleTask();
        
        particleTask = Task.builder()
            .interval(500, TimeUnit.MILLISECONDS)
            .execute(() -> {
                if (!isActive || buildingArea == null) return;
                
                World world = Sponge.getServer().getWorlds().iterator().next();
                for (Vector3i origin : selectedAreas) {
                    showRegionHighlight(world, origin);
                    showLatentCorrespondence(world, origin);
                }
            })
            .submit(plugin);
    }

    private void stopParticleTask() {
        if (particleTask != null) {
            particleTask.cancel();
            particleTask = null;
        }
    }

    private void showRegionHighlight(World world, Vector3i origin) {
        ParticleEffect particleEffect = ParticleEffect.builder()
            .type(currentToolType == ToolType.RAISE ? 
                  ParticleTypes.HAPPY_VILLAGER : // Green particles for raise
                  ParticleTypes.ANGRY_VILLAGER)  // Red particles for lower
            .build();

        // Show particles on all edges of the 4x4x4 cube
        for (int i = 0; i <= 4; i++) {
            // Bottom edges
            spawnParticle(world, origin.getX() + i, origin.getY(), origin.getZ(), particleEffect);
            spawnParticle(world, origin.getX() + i, origin.getY(), origin.getZ() + 4, particleEffect);
            spawnParticle(world, origin.getX(), origin.getY(), origin.getZ() + i, particleEffect);
            spawnParticle(world, origin.getX() + 4, origin.getY(), origin.getZ() + i, particleEffect);
            
            // Top edges
            spawnParticle(world, origin.getX() + i, origin.getY() + 4, origin.getZ(), particleEffect);
            spawnParticle(world, origin.getX() + i, origin.getY() + 4, origin.getZ() + 4, particleEffect);
            spawnParticle(world, origin.getX(), origin.getY() + 4, origin.getZ() + i, particleEffect);
            spawnParticle(world, origin.getX() + 4, origin.getY() + 4, origin.getZ() + i, particleEffect);
            
            // Vertical edges
            spawnParticle(world, origin.getX(), origin.getY() + i, origin.getZ(), particleEffect);
            spawnParticle(world, origin.getX() + 4, origin.getY() + i, origin.getZ(), particleEffect);
            spawnParticle(world, origin.getX(), origin.getY() + i, origin.getZ() + 4, particleEffect);
            spawnParticle(world, origin.getX() + 4, origin.getY() + i, origin.getZ() + 4, particleEffect);
        }
    }

    private void spawnParticle(World world, double x, double y, double z, ParticleEffect effect) {
        world.spawnParticles(effect, new Vector3d(x + 0.5, y + 0.5, z + 0.5));
    }

    public boolean hasSelections() {
        return !selectedAreas.isEmpty();
    }

    // Get a formatted string of all selected indices
    public String getSelectedIndicesString() {
        if (!hasSelections() || buildingArea == null) {
            return "no selections";
        }

        Point outputOrigin = buildingArea.getOutputOrigin();
        StringBuilder indices = new StringBuilder();
        
        for (Vector3i region : selectedAreas) {
            int relX = (region.getX() - outputOrigin.getX()) / 4;
            int relY = (region.getY() - outputOrigin.getY()) / 4;
            int relZ = (region.getZ() - outputOrigin.getZ()) / 4;
            
            if (indices.length() > 0) {
                indices.append(", ");
            }
            indices.append(String.format("[%d,%d,%d]", relX, relY, relZ));
        }
        
        return indices.toString();
    }
}