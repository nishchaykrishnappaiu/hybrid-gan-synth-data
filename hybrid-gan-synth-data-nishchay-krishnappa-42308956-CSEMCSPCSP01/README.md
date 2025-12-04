# hybrid-gan-synth-data-nishchay-krishnappa-42308956-CSEMCSPCSP01
<h1>Problem Statement</h1>
<p>Currently, privacy concerns and the difficulty of locating or gathering high-quality data make it difficult
for researchers and students to obtain enough of it for machine learning projects. When using tabular
datasets for classification jobs, I encountered the same issues myself. This inspired me to figure out
how to provide practical, meaningful data when there isn’t enough.</p>
<br>
<h1>Goals/Requriments</h1>
<ul>
   <li>Generate realistic synthetic tabular data that mirrors real dataset patterns.</li>
   <li>Preserve both categorical and continuous feature relationships.</li>
   <li>Provide an end-to-end automated workflow:</li>
      <ol>
         <li>Preprocessing</li>
         <li>Model Training</li>
         <li>Synthetic Data Generation</li>
         <li>Evaluation & Comparison</li>
      </ol>
   <li>Help students and researchers experiment safely without violating privacy policies.</li>
<h1>Tech Stack</h1>
   <ul>
      <li>Programming Language: Python</li>
      <li>Libraries & Frameworks:</li>
      <ol>
         <li>Pandas, NumPy – Data preprocessing</li>
         <li>CTGAN / SDV – Model training</li>
         <li>Matplotlib, SciPy – Visualization and statistical tests</li>
      </ol>
   </ul>
</ul>
<h1>Phase status</h1>
<p> Development Phase</p>
<h1>Risk Table</h1>
</head>
<body>
  <div class="table-wrap" role="region" aria-label="Risk register table" tabindex="0">
    <table>
      <thead>
        <tr>
          <th scope="col">Type</th>
          <th scope="col">Description</th>
          <th scope="col">Likelihood</th>
          <th scope="col">Impact</th>
          <th scope="col">Mitigation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Resource</strong></td>
          <td>Using datasets that are under copyright or contain private data</td>
          <td><span class="badge">Medium</span></td>
          <td><span class="badge">High</span></td>
          <td>Use only open-source datasets (e.g., UCI, Kaggle public datasets)</td>
        </tr>
        <tr>
          <td><strong>Technical</strong></td>
          <td>Model overfitting or poor-quality synthetic samples</td>
          <td><span class="badge">Medium</span></td>
          <td><span class="badge">Medium</span></td>
          <td>Increase training epochs and fine-tune CTGAN hyperparameters</td>
        </tr>
        <tr>
          <td><strong>Performance</strong></td>
          <td>Long training times on large datasets</td>
          <td><span class="badge">High</span></td>
          <td><span class="badge">Medium</span></td>
          <td>Enable GPU runtime if available; use optimized batch sizes</td>
        </tr>
        <tr>
          <td><strong>Quality</strong></td>
          <td>Statistical difference between real and synthetic data</td>
          <td><span class="badge">Medium</span></td>
          <td><span class="badge">High</span></td>
          <td>Evaluate using KS, L1, and JSD metrics and improve model parameters</td>
        </tr>
      </tbody>
    </table>
    <p class="muted">Tip: You can copy this into any HTML page. Remove the &lt;style&gt; block if you prefer your own CSS.</p>
  </div>
</body>
<h1>Expected Outcome</h1>
<p>A functional Python-Based pipeline that:</p>
<ul>
   <li>Generates synthetic datasets resembling real ones</li>
   <li>Produces comparison reports and plots</li>
   <li>Can be reused for other tabular datasets by simply changing the input CSV</li>
</ul>
