# PortraitLighting

1. **Create a new conda environment**

   ```bash
   conda env create -f environment.yml
   ```

2. **Train a LoRA module**

    Use lora_train.py

3. **Merge the LoRA weights into the FLUX.1-dev model to initialize the Relight model** 

   Use merge/flux_merge_lora.py

4. **Train a Relight model**

   Use relight_train.py

5. **Test the trained model in a gradio demo**

   Use relight_gradio.py