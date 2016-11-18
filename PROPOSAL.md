##Clustered Shading


Clustered shading is a rendering technique similar in concept to tiled-based shading and like tile-based shading, can be applied to deferred and forward shading. In clustered shading, instead of grouping the scene into arbitrary tiles based on screen position, we instead group by similar properties in the scene like normals.  This has been shown to outperform existing tile-based shading techniques in both best-case and worst-case scenarios. Additionally, clustered shading allows for 2-3 times more lights in the scene than tiled-based deferred.  This method has been incorporated into the Avalanche Engine.


I’d like to implement clustered shading in Vulkan because it would be a great opportunity to familiarize myself with the API.  Additionally, it will be an opportunity to learn about how the various features of Vulkan, like pipelines and command buffers, may be used to optimize the rendering pipeline.


Finally, some stretch goals I have in mind include a few additional rendering techniques, some of which will be dependent on whether I end up building a deferred or forward renderer.  The three stretch goals I have in mind are volumetric rendering, area lights, and anti-aliasing. Ubisoft presented a compute shader-based technique for volumetric rendering in 2014 which will be compatible with both types of renderers. One of the techniques presented this year for rendering area lights is using linearly transformed cosines to approximate BRDF’s. Finally, anti-aliasing may require different implementations depending on the type of shader. For the deferred shading, one interesting implementation would be Aggregate G-Buffer Anti-Aliasing.


###Goals:

* Clustered Shader in Vulkan
* Diffuse/Bling-Phong materials
* Port of features from project 5


###Stretch:
* Volumetric compute shaders
* Area lights with LTC
* Anti-aliasing


References:

1. http://www.humus.name/Articles/PracticalClusteredShading.pdf

2. http://www.cse.chalmers.se/~uffe/clustered_shading_preprint.pdf

3. https://eheitzresearch.wordpress.com/415-2/

4. http://advances.realtimerendering.com/s2014/wronski/bwronski_volumetric_fog_siggraph2014.pdf

5. http://advances.realtimerendering.com/s2016/AGAA_UE4_SG2016_6.pdf


