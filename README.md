# VulkanClusteredShader

Xiaomao Ding, University of Pennsylvania, CIS565 Final Project

### Introduction
This code will implement a clustered forward shader using the Vulkan API. For more information about what clustered shading is, refer to the project proposal ([here](https://github.com/xnieamo/VulkanClusteredShader/blob/master/PROPOSAL.md)) for now. I will update this introductionary part as I work on the project.

### Milestones
1. Implement basic forward renderer in Vulkan
2. Implement clustered shading + performance analysis
3. Anything missing from Milestone 2 + stretch goals as listed in the proposal


### Milestone 1
<p align="center">
  <img src="https://github.com/xnieamo/VulkanClusteredShader/blob/master/img/modelExample.gif?raw=true">
</p>

So far, I have completed the Vulkan tutorial and generated the above GIF using the code from the tutorial. I need to refactor the code because it is currently sitting in a single file, which will lead to headaches later. Furthermore, there are no camera controls implemented yet which will also be necessary to build in. Furthermore, the current code is just reading in texture color values and not performing any real rendering calculations.

### References
Base code: The code in this repository starts off from and modifies the code from the online Vulkan tutorial by Alexander Overvoorde.

1. Alexander Overvoorde. [https://vulkan-tutorial.com](https://vulkan-tutorial.com/Introduction).

Libraries used:

1. [http://www.glfw.org/](http://www.glfw.org/)
2. [http://glm.g-truc.net/0.9.8/index.html](http://glm.g-truc.net/0.9.8/index.html)
3. [https://github.com/nothings/stb](https://github.com/nothings/stb)
4. [https://github.com/syoyo/tinyobjloader](https://github.com/syoyo/tinyobjloader)
