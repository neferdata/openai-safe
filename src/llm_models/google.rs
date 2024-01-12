use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::stream::StreamExt;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::constants::{GOOGLE_GEMINI_API_URL, GOOGLE_VERTEX_API_URL};
use crate::{
    domain::{GoogleGeminiProApiResp, RateLimit},
    llm_models::LLMModel,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
//Google docs: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
pub enum GoogleModels {
    GeminiPro,
    GeminiProVertex,
}

#[async_trait(?Send)]
impl LLMModel for GoogleModels {
    fn as_str(&self) -> &'static str {
        match self {
            GoogleModels::GeminiPro | GoogleModels::GeminiProVertex => "gemini-pro",
        }
    }

    fn default_max_tokens(&self) -> usize {
        //https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
        match self {
            GoogleModels::GeminiPro | GoogleModels::GeminiProVertex => 32_000,
        }
    }

    fn get_endpoint(&self) -> String {
        //The URL requires GOOGLE_REGION and GOOGLE_PROJECT_ID env variables defined to work.
        //If not set GOOGLE_REGION will default to 'us-central1' but GOOGLE_PROJECT_ID needs to be defined.
        match self {
            GoogleModels::GeminiPro => GOOGLE_GEMINI_API_URL.to_string(),
            GoogleModels::GeminiProVertex => GOOGLE_VERTEX_API_URL.to_string(),
        }
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        _max_tokens: &usize,
        temperature: &u32,
    ) -> serde_json::Value {
        //Prepare the 'messages' part of the body
        let base_instructions_json = json!({
            "text": self.get_base_instructions(Some(function_call))
        });

        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
        let output_instructions_json =
            json!({ "text": format!("'Output Json schema': {schema_string}") });

        let user_instructions_json = json!({
            "text": instructions,
        });

        let contents = json!({
            "role": "user",
            "parts": vec![
                base_instructions_json,
                output_instructions_json,
                user_instructions_json,
            ],
        });

        let generation_config = json!({
            "temperature": temperature,
        });

        json!({
            "contents": contents,
            "generationConfig": generation_config,
        })
    }
    /*
     * This function leverages Mistral API to perform any query as per the provided body.
     *
     * It returns a String the Response object that needs to be parsed based on the self.model.
     */
    async fn call_api(
        &self,
        api_key: &str,
        body: &serde_json::Value,
        debug: bool,
    ) -> Result<String> {
        //Get the API url
        let model_url = self.get_endpoint();

        //Make the API call
        let client = Client::new();

        //Send request
        let response = client
            .post(model_url)
            .header(header::CONTENT_TYPE, "application/json")
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await?;

        // Google API streams the results. We need to handle that
        // Check if the API uses streaming
        if response.status().is_success() {
            let mut stream = response.bytes_stream();
            let mut streamed_response = String::new();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;

                // Convert the chunk (Bytes) to a String
                let mut chunk_str = String::from_utf8(chunk.to_vec()).map_err(|e| anyhow!(e))?;

                // The chunk response starts with "data: " that needs to be remove
                if chunk_str.starts_with("data: ") {
                    // Remove the first 6 characters ("data: ")
                    chunk_str = chunk_str[6..].to_string();
                }

                //Convert response chunk to struct representing expected response format
                let gemini_response: GoogleGeminiProApiResp = serde_json::from_str(&chunk_str)?;

                //Extract the data part from the response
                let part_text = gemini_response
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.content.role.as_deref() == Some("model"))
                    .flat_map(|candidate| &candidate.content.parts)
                    .map(|part| &part.text)
                    .fold(String::new(), |mut acc, text| {
                        acc.push_str(text);
                        acc
                    });

                //Add the chunk response to output string
                streamed_response.push_str(&part_text);

                // Debug log each chunk if needed
                if debug {
                    info!(
                        "[debug][Google Gemini] Received response chunk: {:?}",
                        chunk
                    );
                }
            }

            Ok(streamed_response)
        } else {
            let response_status = response.status();
            let response_txt = response.text().await?;
            Err(anyhow!(
                "[allms][Google][{}] Response body: {:#?}",
                response_status,
                response_txt
            )
            .into())
        }
    }

    //Because GeminPro streams data in chunks the extraction of data/text is handled in call_api method. Here we only pass the input forward
    fn get_data(&self, response_text: &str, _function_call: bool) -> Result<String> {
        Ok(response_text.to_string())
    }

    //This function allows to check the rate limits for different models
    fn get_rate_limit(&self) -> RateLimit {
        //https://ai.google.dev/models/gemini
        RateLimit {
            tpm: 60 * 32_000, // Google only specifies RPM. TPM is calculated from that
            rpm: 60,
        }
    }
}
