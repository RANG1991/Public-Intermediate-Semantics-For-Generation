<!-- Intro-->

<!--
* Thanks for reviewing my Project-README-Template! 
* 
* Read the comments for an easy step by step guide. Enjoy!
-->

<!-- Logo Section  --> <!-- Required -->

<!--
* Insert your github profile URL in the <a> "href" attribute bellow (line-25)
* 
* Insert an image URL in the <img> "src" attribute bellow. (line-26)
-->
<div align="center">
    <a href="https://github.com/RANG1991" target="_blank">
    </a>
</div>


<!-- Project title 
* use a dynamic typing-SvG here https://readme-typing-svg.demolab.com/demo/
*
*  Instead you can type your project name after a # header
-->

<div align="center">
    <h1>Generating Intermediate Representations for Compositional Text-To-Image Generation</h1>
</div>

<div align="center">
    <h2>About</h2>
</div>
<!-- 
* information about the project 
* 
* keep it short and sweet
-->
<div align="center">
For this project, we investigated the use of intermediate representations (e.g. segmentation maps and depth maps) to enhance the performance of Stable Diffusion model. More specifically, we operated in two stages, in the first stage using Stable Diffusion to generate an intermediate representation(s), and in the second stage using ControlNet to produce an image based on these representation(s).
</div>

</br>
<div align="center">
  <p><em>Qualitative results.</em></p>
  <img src="https://github.com/RANG1991/Public-Intermediate-Semantics-For-Generation/blob/main/static/images/Results_Figure_separate_controls.png" 
  alt="qualitative_results" 
  style="width:700px;height:500px;">
</div>

</br>
<div align="center">
  <p><em>Quantitative results.</em></p>
  <img src="https://github.com/RANG1991/Public-Intermediate-Semantics-For-Generation/blob/main/static/images/Results_FID_CLIP.png" 
  alt="quantative_results" 
  style="width:700px;height:500px;">
</div>

</br>
<div align="center">
  <p><em>The pipeline of our model. In the first step, we used Stable Diffusion to generate single or multiple aligned representation(s). In the second step, we used these representation(s) to generate the final image.</em></p>
  <img src="https://github.com/RANG1991/Public-Intermediate-Semantics-For-Generation/blob/main/static/images/Pipeline_Figure_separate_controls.png" 
  alt="Pipeline_model" 
  style="width:800px;height:500px;">
</div>

</br>
<div align="center">
  <p><em>The pipeline of our alignment model. In order to align two representations together, we interleaved two U-Nets from two pre-trained Stable Diffusion models.</em></p>
  <img src="https://github.com/RANG1991/Public-Intermediate-Semantics-For-Generation/blob/main/static/images/Pipeline_Figure_2_controls_detailed.png" 
  alt="Pipeline_alignment_model" 
  style="width:800px;height:500px;">
</div>

<div align="center">
    <h2>Acknowledgments</h2>
    We used the following resources in our research:
    </br>
    LanguageBind - https://github.com/PKU-YuanGroup/LanguageBind
    </br>
    Vision and Language Transformer (VITL) - https://github.com/dandelin/ViLT
    </br>
    Depth Anything - https://github.com/LiheYoung/Depth-Anything
    </br>
    ControlNet - https://github.com/lllyasviel/ControlNet, https://huggingface.co/thibaud/controlnet-sd21
</div>
