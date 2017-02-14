# ai-neural-network-project
bu ai neural network 2-layer and 3-layer nn

<html>
<head>
<title> CS440/640 Homework Template: PROG3 Student Name Sang-Joon Lee  </title>
<style>
<!--
body{
font-family: 'Trebuchet MS', Verdana;
}
p{
font-family: 'Trebuchet MS', Times;
margin: 10px 10px 15px 20px;
}
h5{
margin: 10px 0px 0px 100px;

}
    
h4{
margin: 10px 0px 0px 50px;
}

h3{
margin: 10px;
}
h2{
margin: 10px;
}
h1{
margin: 10px 0px 0px 20px;
}
div.main-body{
align:center;
margin: 30px;
}
hr{
margin:20px 0px 20px 0px;
}
.main-body img{
    width: 50%;
    height: auto;
    margin-left: auto;
    margin-right: auto;
    display: block;
}
img:parent{
    text-align: center;
}
    
-->
</style>
</head>

<body>
<center>
<a href="http://www.bu.edu"><img border="0" src="http://www.cs.bu.edu/fac/betke/images/bu-logo.gif"
width="119" height="120"></a>
</center>

<h1>Hidden Markov Model</h1>
<p> 
 CS 440/640 P3 <br>
 Tyrone Hou, Sang-Joon Lee, Srivathsa Rajagopal, Huai-Chun (Daniel) Shih <br>
    4/05/16
</p>

<div class="main-body">
<hr>
<h2> Problem Definition </h2>
<p>
 A hidden Markov model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. 
    Hidden Markov models are especially known for their application in temporal pattern recognition such as speech, handwriting, gesture recognition, and bioinformatics.
</p>
    
<p>    
The objective of this work is to understand how Hidden Markov Model (HMM) works by implementing a basic English sentence recognizer. The English sentence recognizer system is able to recognize very limited vocabulary of words. This work is implemented using python.
</p>    
        
<h3> Hidden Markov Model </h3>
    <p>
        An HMM was defined based on a simplified version of English. This version has only very limited vocabulary and does not use articles or prepositions. Only the following words are included in the vocabulary supported by the HMM: 
    </p>
    <div align="center">
        <p>
            {"kids", "robots", "do", "can", "play", "eat", "chess", "food"}
        </p>
    </div>

    <p>
        The HMM is expected to recognize and parse sentences that use the above vocabulary. The sentences:
    </p>
    <div align="center">
        <p>
            "kids play chess" <br>
            "can robots eat food"
        </p>
    </div>

    
    <h4>File Format to Describe a HMM</h4>
        <p  style="text-indent:30px">
            The format of an .hmm file is as follows:  
        </p>
        <p  style="text-indent:30px">
            The first line contains integers N (number of states), M (number of observation symbols), and T (number of time steps or length of oberservation sequences).  
        </p>
        <p style="text-indent:30px">
            The second contains four strings: 
        </p>
        <div align="center">
            <p>
                SUBJECT AUXILIARY PREDICATE OBJECT
            </p>
        </div>
        <p style="text-indent:30px">
            which refer to four basic English syntactic structures. Each is used to name an individual HMM state.
        </p>
    
        <p style="text-indent:30px">
            The third line contains strings:
        </p>
        <div align="center">
            <p>
                kids robots do can play eat chess food
            </p>
        </div>

        <p style="text-indent:30px">
            that provide the vocabulary to be used in the observation sentences. 
        </p>
    
        <p style="text-indent:30px">
            Then comes a line with the text "a:", followed by the matrix a.   The matrix b and vector pi are similarly represented.  The matrix and vector elements are floating-point numbers less than or equal to 1.0.
        </p>

        <p style="text-indent:30px">
            HMM should be in the correct form, which means the percentages across rows add up to 1.0. Moreover, the HMM and a set of observation data use the same finite alphabet of observation symbols. 
        </p>

    <h4>Example HMM File</h4>

    <p style="text-indent:30px">
        A hidden markov model is provided in a text file as a <a href="http://www.cs.bu.edu/fac/betke/cs440/restricted/p3/sentence.hmm">.hmm </a> file.                 
    </p>
    
    <p style="text-indent:30px">  
        The following is a graphical representation of the hidden markov model for the .hmm file provided above.
    </p>
   <p class="one" style="text-indent:5em"> 
            <img src="img/given_hmm_model.jpg">         
   </p>  
    
<h3> Observations </h3>
    <h4>File Format to Describe an Observation Sequence</h4>
        <p style="text-indent:30px">
            The format of a .obs file is as follows:  
        </p>
        <p style="text-indent:100px">
        The first line of the file contains the number of data sets in the file.  For each data set, the number of observations in the set appears on a line by itself. 
        </p>
        <p style="text-indent:100px">
        The next line are the observations, composed by the tokens from the vocabulary.   
        </p>        
        <p style="text-indent:30px">
        Here are two sample files: <a href="http://www.cs.bu.edu/fac/betke/cs440/restricted/p3/example1.obs">
        example1.obs</a> and 
            <a href="http://www.cs.bu.edu/fac/betke/cs440/restricted/p3/example2.obs">
        example2.obs</a> file. 
    </p>
<hr>
<h2> Method and Implementation </h2>
<h3> 1. Pattern Recognition </h3>
    <p>
        The pattern recognition of English sentence was implemented using the "forward part" of the forward/backward procedure. Given the HMM and one or more observation sequence, the forward procedure calculates and reports the observation probability of each input sequence. The following are the steps for "forward procedure":
    </p>
    <h4> 1) Initialization </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/forward_init.PNG">         
        </p>   
    <h4> 2) Induction </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/forward_induction.PNG">         
        </p> 
    <h4> 3) Termination </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/forward_termination.PNG">         
        </p> 
<h3> 2. State-Path Determination </h3>
    <p>    
        We implemented the state-path determination of Hidden Markov Model using the Viterbi algorithm. The Viterbi algorithm is used to determine the optimal state path for each observation set from the input file. For each observation set, its probability is reported. 
    </p>
    
    <p>
        The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states. This is called state path or viterbi path, which is a sequence of observed events.  The following are the steps for Viterbi algorithm:
    </p>
    <h4> 1) Initialization </h4>
        <p style="text-indent:5em"> 
            <img src="img/viterbi_step_init.PNG">         
        </p>               
    <h4> 2) Recursion </h4>
        <p style="text-indent:5em"> 
            <img src="img/viterbi_step_recursion.PNG">         
        </p>               
    <h4> 3) Termination </h4>    
        <p style="text-indent:5em"> 
            <img src="img/viterbi_step_termination.PNG">         
        </p>
    <h4> 4) Path (State Sequence) Backtracking: </h4>   
        <p style="text-indent:5em"> 
            <img src="img/viterbi_step_backtrack.PNG">         
        </p>

<h3> 3. Model Optimization </h3>
    <p>    
        We implemented optimizaes the HMM using one iteration of the Baum-Welch algorithm.  After all data sets are processed, optimized HMM is saved in a new file.  For each data set, optimize prints out P(O | lambda) for the old HMM before optimization, and P(O | lambda) for the new HMM after optimization. The following is steps to Baum-Welch algorithm:
    </p>
    <h4> 1) Forward Procedure </h4>
    <p style="text-indent:30px">
        The "forward procedure" is similar to pattern recognition method described in previous section. 
    </p>
    <h4> 2) Backward Procedure </h4>
        <h5> A) Initialization </h5>
        <p class="one" style="text-indent:5em"> 
            <img src="img/backward_init.PNG">         
        </p>   
        <h5> B) Induction </h5>
        <p class="one" style="text-indent:5em"> 
            <img src="img/backward_induction.PNG">         
        </p>   
    <h4> 3) Calculate Xi </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/optimize_xi.PNG">         
        </p>  
    <h4> 4) Calculate Gamma </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/optimize_gamma_t.PNG">         
        </p>  
    <h4> 5) Update </h4>
        <p class="one" style="text-indent:5em"> 
            <img src="img/optimze_update.PNG">         
        </p> 
<hr>
<h2>Experiments</h2>
<p>
For our experiments, we used three example observation files with varying number of observations and . 
<br><br>
    
    <p style="text-indent:30px">
        Here are two sample files: <a href="http://www.cs.bu.edu/fac/betke/cs440/restricted/p3/example1.obs">
        example1.obs</a> and 
            <a href="http://www.cs.bu.edu/fac/betke/cs440/restricted/p3/example2.obs">
        example2.obs</a> file. 
    </p>
    
    <h2>Example1.obs</h2>
        <p class="one" style="text-indent:5em"> 
            <img src="img/observation_example1.PNG">         
        </p>   
    <h2>Example2.obs</h2>
        <p class="one" style="text-indent:5em"> 
            <img src="img/observation_example2.PNG">         
        </p>       
    <h2>Example3.obs</h2>
        <p class="one" style="text-indent:5em"> 
            <img src="img/observation_example3.PNG">         
        </p>    
    <h2>Example4.obs</h2>
        <p>
             You can find the exmaple file for<a href="examples/example4.obs">
        example4.obs</a>.
        </p>
<hr>
<h2> Results</h2>
<p> 
    The following are output of the each of the observations described in experiment section. The output of input observations for 1) Recognization, 2) Statepath Determination, and 3) Optimization as shown below.
</p>
<h3> 1. Recognization </h3>
    <p>
        The following commandline output shows the recognition probabilty of the observations example1.obs, example2.obs and example3.obs.
    </p>
        <p style="text-indent:5em"> 
            <img src="img/recognize.png">         
        </p>
<h3> 2. Statepath </h3>
    <p>
        The following commandline output shows the observation probabilty and statepath output of the observations example1.obs, example2.obs and example3.obs.
    </p>
        <p style="text-indent:5em"> 
            <img src="img/statepath.png">         
        </p>
<h3> 3. Optimization </h3>
    <p>
        The following commandline output shows the observation probabilty of prior and post optimization of HMM using Baum-Welch method for the observations example1.obs, example2.obs and example3.obs.
</p>
        <p style="text-indent:5em"> 
            <img src="img/optimize.png">         
        </p>
<h4> 3.1 Optimzed HMM - Graphical Representation </h4>    
    <p style="text-indent:50px">
        The following figure illustrates the Hidden Markov Model after optimization. 
    </p>
    <p style="text-indent:5em"> 
        <img src="img/optimized_hmm_model.jpg">    
    </p>
        
    <hr>
        <h2> Discussion </h2>
    <br>
    <h3>Pattern Recognition </h3>
    
    <h4>Question: For the current application, why does this probability seem lower than we expect? What does this probability tell you? Does the current HMM always give a reasonable answer? For instance, what is the output probability for the below sentence?
<br><br>
"robots do kids play chess"
"chess eat play kids"
<br><br>
    </h4>
    
    <div style="text-indent:5em">
    <p style="text-indent:5em">
    For a given sequence of length T, the probability space of that sequence is all sequences of length T. In other words, if the probabilities of all possible T length sequences are summed we should get a value of 1.0. Since loops exist between certain states, the total possible number of valid sequences is high for values of T greater than 2, and thus the probability of any one sequence occurring is unlikely to be very high. Additionally, the Hidden Markov Model has not been trained to recognize any particular sequence; it's transition and output matrices may not perfectly reflect English sentences. This said, the HMM seems to produce reasonable syntactic structures. Nonsensical sentences like "robots do kids play chess" and "chess eat play kids" have low probabilities. Due to its limited vocabulary, the system sometimes produces sentences with decent structure but absurd meanings. For example, the three word sequence with the highest probability is "kids eat food", which makes sense both structurally and semantically. However, "kids play food" has the same probability of occurring. "can kids play" and "do kids eat" both have (relatively) high probabilities at 0.01575, yet according to the model "kids can kids" and "kids do kids" both have a higher chance of occurring (0.021).
    </p>
    </div>
    
    
<hr>
    <h3>StatePath</h3>

    <h4>Question: What can we tell from the reported optimal path for syntax analysis purpose? Can the HMM always correctly distinguish "statement" from "question" sentence? Why?    
    </h4>    
    
    <p style="text-indent:5em">
        The reported optimal path allows us to distinguish between whether an observation sequence is a statement or a question. The original model is able to do so with great accuracy. With our limited vocabulary, the only way to start a question is with the word 'can' or the word 'do'. Since both only have non-zero output probabilities in the auxiliary state, we can easily say a sentence is a question if it starts in the auxiliary state.
    </p>
    
<hr>    
    <h3>Opimization</h3>
        <h4> Why should you not try to optimize an HMM with zero observation probability?
        </h4>
     
    <p  style="text-indent:5em">
When the observation probability is zero, then there won't be any output from that state. Additionally, the observation probability is used as a normalization factor.
    </p>
 
    <p  style="text-indent:5em">
This causes the divide by zero errors in our &gamma; (gamma) calculation and subsequently in the &xi; (xi) calculation. This error if not handled leads to unexpected and incorrect results when updating our transition probability matrices.
    </p>

<hr>
<h3>Model Enhancement</h3>

<h4>Now supposed you want this HMM to model new syntax structures, like "PRESENT TENSE" or "ADVERB," so that the following sentences can be parsed: 
    <br><br>"robots can play chess well"
    <br>"kids do eat food fast"
    <br><br>
    Question:  What kinds of changes will you need to make in the above HMM?Please describe your solution with an example of the modified matrices a, b and pi in the submitted web page.
</h4>    

    <p  style="text-indent:5em">
       In order to model new syntax structures we must expand the transition, observation, and pi matrices. For example, we can add the state "ADVERB" with new observations "fast" and "well". The a, b, and pi matrices would then look like the following:
    </p>

    <p style="text-indent:5em"> 
        <img src="img/question4_output.png">    
    </p>

<hr>
<h2> Conclusions </h2>

    <p>
        In this work, we have illustrated application of Hidden Markov Model to recognize English sentences. The behaviour and properties of Hidden Markov Model was investigated and studied. The recognition algorithm, statepath determination using Viterbi algorithm and learning method using Baum-Welch algorithm were implemented and results were presented.  This work illustrates that Hidden Markov Model works well for simple English sentence recognizer and very possible to be extended to a large recognition model.   
    </p>


<hr>
<h2> Credits and Bibliography </h2>
    
<p> Wikipedia - Hidden Markov Model:  https://en.wikipedia.org/wiki/Hidden_Markov_model </p>
<p> Wikipedia - Baum Welch: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm </p>
    <p> A tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, Lawrence R. Rabiner: http://www.cs.bu.edu/fac/betke/cs440/restricted/papers/rabiner.pdf
    </p>

    
<hr>
</div>
</body>



</html>
