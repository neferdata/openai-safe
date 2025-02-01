use anyhow::{anyhow, Result};
use jsonschema::JSONSchema;
use log::error;
use log::info;
use reqwest::Client;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time;
use tokio::time::timeout;

use crate::assistants::{OpenAIAssistantResource, OpenAIAssistantVersion, OpenAIVectorStore};
use crate::constants::OPENAI_ASSISTANT_INSTRUCTIONS;
use crate::domain::{
    AllmsError, OpenAIAssistantResp, OpenAIMessageListResp, OpenAIMessageResp, OpenAIRunResp,
    OpenAIThreadResp,
};
use crate::enums::{OpenAIAssistantRole, OpenAIRunStatus};
use crate::llm_models::{LLMModel, OpenAIModels};
use crate::utils::get_type_schema;

/// [OpenAI Docs](https://platform.openai.com/docs/assistants/overview)
///
/// The Assistants API allows you to build AI assistants within your own applications.
/// An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries.
/// The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling.
/// In the future, we plan to release more OpenAI-built tools, and allow you to provide
/// your own tools on our platform.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OpenAIAssistant {
    id: Option<String>,
    thread_id: Option<String>,
    run_id: Option<String>,
    model: OpenAIModels,
    instructions: String,
    debug: bool,
    api_key: String,
    version: OpenAIAssistantVersion,
    vector_store: Option<OpenAIVectorStore>,
    temperature: f32,
}

impl OpenAIAssistant {
    /// Constructor of the Assistant
    /// Reasoning models are not currently supported in Assistants API. If requested defaulting to `Gpt4o`
    pub fn new(model: OpenAIModels, open_ai_key: &str) -> Self {
        let model = match model {
            OpenAIModels::O1Preview
            | OpenAIModels::O1Mini
            | OpenAIModels::O1
            | OpenAIModels::O3Mini => OpenAIModels::Gpt4o,
            _ => model,
        };

        OpenAIAssistant {
            id: None,
            thread_id: None,
            run_id: None,
            temperature: model.get_default_temperature(),
            model,
            instructions: OPENAI_ASSISTANT_INSTRUCTIONS.to_string(),
            debug: false,
            api_key: open_ai_key.to_string(),
            // Defaulting to V1 for now
            version: OpenAIAssistantVersion::V1,
            vector_store: None,
        }
    }

    ///
    /// This method can be used to turn on debug mode for the Assistant
    ///
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    ///
    /// This method can be used to set the version of Assistants API Beta
    /// Current default is V1
    ///
    pub fn version(mut self, version: OpenAIAssistantVersion) -> Self {
        self.version = version;
        self
    }

    ///
    /// This method can be used to define the model temperature used by the Assistant
    /// This method accepts % target of the acceptable range for the model
    ///
    pub fn temperature(mut self, temp_target: u32) -> Self {
        self.temperature = self.model.get_normalized_temperature(temp_target);
        self
    }

    ///
    /// This method can be used to define the model temperature used by the Assistant
    /// Using this method the temperature can be set directly without any validation of the range accepted by the model
    /// For a range-safe implementation please consider using `OpenAIAssistant::temperature` method
    ///
    pub fn temperature_unchecked(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /*
     * This function creates an Assistant and updates the ID of the OpenAIAssistant struct
     */
    async fn create_assistant(&mut self) -> Result<()> {
        //Get the assistant API url
        let assistant_url = self
            .version
            .get_endpoint(&OpenAIAssistantResource::Assistants);

        //Get the version-specific header
        let version_headers = self.version.get_headers(&self.api_key);

        let mut assistant_body = json!({
            "instructions": self.instructions.clone(),
            "model": self.model.as_str(),
            "temperature": self.temperature,
        });

        //Get the retrieval / file_search part of the payload (if supported)
        if self.model.tools_support() {
            if let Some(assistant_body_object) = assistant_body.as_object_mut() {
                let tools_payload = self.version.get_tools_payload();
                assistant_body_object.insert("tools".to_string(), tools_payload);
            }
        }

        //Make the API call
        let client = Client::new();

        let response = client
            .post(assistant_url)
            .headers(version_headers)
            .json(&assistant_body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Assistant API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Assistant object
        let response_deser: OpenAIAssistantResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Assistant API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        //Add correct ID to self
        self.id = Some(response_deser.id);

        Ok(())
    }

    ///
    /// This function performs all the orchestration needed to submit a prompt and get and answer
    ///
    pub async fn get_answer<T: JsonSchema + DeserializeOwned>(
        &mut self,
        message: &str,
        file_ids: &[String],
    ) -> Result<T> {
        // Instruct the Assistant to answer with the right Json format
        // Output schema is extracted from the type parameter
        let schema_string = get_type_schema::<T>()?;

        // Call assistant
        let assistant_response = self
            .call_assistant(&schema_string, message, file_ids)
            .await?;

        // Deserialize assistant message
        serde_json::from_str::<T>(&assistant_response).map_err(|e| {
            let error = AllmsError {
                crate_name: "alms".to_string(),
                module: "assistants::openai_assistant".to_string(),
                error_message: format!("Deserialization error: {:?}", e),
                error_detail: assistant_response,
            };
            anyhow!("{:?}", error)
        })
    }

    ///
    /// This function is similar to _get_answer_ however it returns a Json Value matching the provided schema
    ///
    pub async fn get_json_answer(
        &mut self,
        message: &str,
        json_schema: &str,
        file_ids: &[String],
    ) -> Result<Value> {
        // Call assistant
        let assistant_response = self.call_assistant(json_schema, message, file_ids).await?;

        // Deserialize assistant message
        self.get_valid_json(json_schema, &assistant_response)
    }

    // This function performs orchestration with Assistants API to get a message with response
    async fn call_assistant(
        &mut self,
        json_schema: &str,
        message: &str,
        file_ids: &[String],
    ) -> Result<String> {
        // If the assistant and thread are not initialized we do that first
        if self.id.is_none() {
            //Call OpenAI API to get an ID for the assistant
            self.create_assistant().await?;

            //Add first message thus initializing the thread
            self.add_message(OPENAI_ASSISTANT_INSTRUCTIONS, &Vec::new())
                .await?;
        }

        // Instruct Assistant to answer with that schema
        let schema_message = format!(
            "Response should include only the data portion of a Json formatted as per the following schema: {}. 
            The response should only include well-formatted data, and not the schema itself.
            Do not include any other words or characters, including the word 'json'. Only respond with the data. 
            You need to validate the Json before returning.",
            json_schema
        );
        self.add_message(&schema_message, &Vec::new()).await?;

        //Step 2: Add user message and files to thread
        self.add_message(message, file_ids).await?;

        //Step 3: Kick off processing (aka Run)
        self.start_run().await?;

        //Step 4: Check in on the status of the run
        let operation_timeout = Duration::from_secs(600); // Timeout for the whole operation
        let poll_interval = Duration::from_secs(10);

        let _result = timeout(operation_timeout, async {
            let mut interval = time::interval(poll_interval);
            loop {
                interval.tick().await; // Wait for the next interval tick
                match self.get_run_status().await {
                    Ok(resp) => match resp.status {
                        //Completed successfully. Time to get results.
                        OpenAIRunStatus::Completed => {
                            break Ok(());
                        }
                        //TODO: We will need better handling of requires_action
                        OpenAIRunStatus::RequiresAction
                        | OpenAIRunStatus::Cancelling
                        | OpenAIRunStatus::Cancelled
                        | OpenAIRunStatus::Failed
                        | OpenAIRunStatus::Expired => {
                            return Err(anyhow!("Failed to validate status of the run"));
                        }
                        _ => continue, // Keep polling if in_progress or queued
                    },
                    Err(e) => return Err(e), // Break on error
                }
            }
        })
        .await?;

        //Step 5: Get all messages posted on the thread. This should now include response from the Assistant
        let messages = self.get_message_thread().await?;

        messages
            .iter()
            .filter(|message| message.role == OpenAIAssistantRole::Assistant)
            .find_map(|message| {
                message.content.iter().find_map(|content| {
                    content
                        .text
                        .as_ref()
                        .map(|text| self.model.sanitize_json_response(&text.value))
                })
            })
            .ok_or_else(|| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: "No valid response from OpenAI Assistant found.".to_string(),
                    error_detail: format!("{:?}", &messages),
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })
    }

    // This function checks if a Json object matches the schema
    fn get_valid_json(&self, schema: &str, value: &str) -> Result<Value> {
        let schema_value = serde_json::from_str(schema).map_err(|e| {
            let error = AllmsError {
                crate_name: "alms".to_string(),
                module: "assistants::openai_assistant".to_string(),
                error_message: format!("Json Schema parsing error: {:?}", e),
                error_detail: format!("Schema: {:?}", schema),
            };
            anyhow!("{:?}", error)
        })?;

        let compiled_schema = JSONSchema::compile(&schema_value).map_err(|e| {
            let error = AllmsError {
                crate_name: "alms".to_string(),
                module: "assistants::openai_assistant".to_string(),
                error_message: format!("Json Schema compilation error: {:?}", e),
                error_detail: format!("Schema: {:?}", schema_value),
            };
            anyhow!("{:?}", error)
        })?;

        let data_value = serde_json::from_str(value).map_err(|e| {
            let error = AllmsError {
                crate_name: "alms".to_string(),
                module: "assistants::openai_assistant".to_string(),
                error_message: format!("Json data parsing error: {:?}", e),
                error_detail: format!("Data: {:?}", value),
            };
            anyhow!("{:?}", error)
        })?;

        compiled_schema.validate(&data_value).map_err(|_| {
            let error = AllmsError {
                crate_name: "alms".to_string(),
                module: "assistants::openai_assistant".to_string(),
                error_message: "Json Schema validation error".to_string(),
                error_detail: format!("Data: {:?}\nSchema: {:?}", &data_value, &schema_value),
            };
            anyhow!("{:?}", error)
        })?;

        Ok(data_value)
    }

    ///
    /// This method can be used to provide data that will be used as context for the prompt.
    /// Using this function you can provide multiple sets of context data by calling it multiple times. New values will be as messages to the thread
    /// It accepts any struct that implements the Serialize trait.
    ///
    pub async fn set_context<T: Serialize>(mut self, dataset_name: &str, data: &T) -> Result<Self> {
        // If the assistant and thread are not initialized we do that first
        if self.id.is_none() {
            //Call OpenAI API to get an ID for the assistant
            self.create_assistant().await?;

            //Add first message thus initializing the thread
            self.add_message(OPENAI_ASSISTANT_INSTRUCTIONS, &Vec::new())
                .await?;
        }

        let serialized_data = if let Ok(json) = serde_json::to_string(&data) {
            json
        } else {
            return Err(anyhow!("Unable serialize provided input data."));
        };
        let message = format!("'{dataset_name}'= {serialized_data}");
        let file_ids = Vec::new();
        self.add_message(&message, &file_ids).await?;
        Ok(self)
    }

    /*
     * This function creates a Thread and updates the thread_id of the OpenAIAssistant struct
     */
    async fn add_message(&mut self, message: &str, file_ids: &[String]) -> Result<()> {
        //Prepare the body that is to be send to OpenAI APIs
        let mut message = json!({
            "role": "user",
            "content": message.to_string(),
        });

        if !file_ids.is_empty() {
            message = self.version.add_message_attachments(&message, file_ids);
        }

        //If there is no thread_id we need to create one
        match self.thread_id {
            None => {
                let body = json!({
                    "messages": vec![message],
                });

                self.create_thread(&body).await
            }
            Some(_) => self.add_message_thread(&message).await,
        }
    }

    /*
     * This function creates a Thread and updates the thread_id of the OpenAIAssistant struct
     */
    async fn create_thread(&mut self, body: &serde_json::Value) -> Result<()> {
        //Get version-specific URL
        let thread_url = self.version.get_endpoint(&OpenAIAssistantResource::Threads);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        //Make the API call
        let client = Client::new();

        let response = client
            .post(thread_url)
            .headers(version_headers)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Threads API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Thread object
        let response_deser: OpenAIThreadResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Thread API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        //Add thread_id to self
        self.thread_id = Some(response_deser.id);

        Ok(())
    }

    /*
     * This function adds a message to an existing thread
     */
    async fn add_message_thread(&self, body: &serde_json::Value) -> Result<()> {
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        //Get version-specific URL
        let messages_resource = OpenAIAssistantResource::Messages {
            thread_id: self.thread_id.clone().unwrap_or_default(),
        };
        let message_url = self.version.get_endpoint(&messages_resource);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        //Make the API call
        let client = Client::new();

        let response = client
            .post(message_url)
            .headers(version_headers)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let _response_deser: OpenAIMessageResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Messages API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        Ok(())
    }

    /*
     * This function gets all message posted to an existing thread
     */
    async fn get_message_thread(&self) -> Result<Vec<OpenAIMessageResp>> {
        if self.thread_id.is_none() {
            return Err(anyhow!("No active thread detected."));
        }

        //Get version-specific URL
        let message_resource = OpenAIAssistantResource::Messages {
            thread_id: self.thread_id.clone().unwrap_or_default(),
        };
        let message_url = self.version.get_endpoint(&message_resource);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        //Make the API call
        let client = Client::new();

        let response = client
            .get(message_url)
            .headers(version_headers)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into a vector of OpenAIMessageResp objects
        let response_deser: OpenAIMessageListResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Messages API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        Ok(response_deser.data)
    }

    /*
     * This function starts an assistant run
     */
    async fn start_run(&mut self) -> Result<()> {
        let assistant_id = if let Some(id) = self.id.clone() {
            id
        } else {
            return Err(anyhow!("No active assistant detected."));
        };

        let thread_id = if let Some(id) = self.thread_id.clone() {
            id
        } else {
            return Err(anyhow!("No active thread detected."));
        };

        //Get version-specific URL
        let run_resource = OpenAIAssistantResource::Runs { thread_id };
        let run_url = self.version.get_endpoint(&run_resource);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        let body = json!({
            "assistant_id": assistant_id,
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(run_url)
            .headers(version_headers)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Messages API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIRunResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Run API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        //Update run_id
        self.run_id = Some(response_deser.id);

        Ok(())
    }

    /*
     * This function checks the status of an assistant run
     */
    async fn get_run_status(&self) -> Result<OpenAIRunResp> {
        let thread_id = if let Some(id) = self.thread_id.clone() {
            id
        } else {
            return Err(anyhow!("No active thread detected."));
        };

        let run_id = if let Some(id) = self.run_id.clone() {
            id
        } else {
            return Err(anyhow!("No active run detected."));
        };

        //Get version-specific URL
        let run_resource = OpenAIAssistantResource::Run { thread_id, run_id };
        let run_url = self.version.get_endpoint(&run_resource);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        //Make the API call
        let client = Client::new();

        let response = client.get(run_url).headers(version_headers).send().await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Run status API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Message object to confirm if there were any errors
        let response_deser: OpenAIRunResp =
            serde_json::from_str(&response_text).map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!("Run API response serialization error: {}", error),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })?;

        Ok(response_deser)
    }

    ///
    /// This method can be used to attach a Vector Store object to an Assistant
    ///
    pub async fn vector_store(&mut self, vector_store: OpenAIVectorStore) -> Result<Self> {
        if self.version == OpenAIAssistantVersion::V1 {
            return Err(anyhow!(
                "[OpenAI][Assistants] OpenAI Assistants API v1 does not support Vector Store."
            ));
        }
        if vector_store.id.is_none() {
            return Err(anyhow!(
                "[OpenAI][Assistants] Unable to attach Vector Store. No valid ID found."
            ));
        }
        self.attach_vector_store(&vector_store).await?;
        self.vector_store = Some(vector_store);
        Ok(self.clone())
    }

    /*
     * This function attaches a vector store to an Assistant
     */
    async fn attach_vector_store(&mut self, vector_store: &OpenAIVectorStore) -> Result<()> {
        // If the assistant and thread are not initialized we do that first
        if self.id.is_none() {
            //Call OpenAI API to get an ID for the assistant
            self.create_assistant().await?;

            //Add first message thus initializing the thread
            self.add_message(OPENAI_ASSISTANT_INSTRUCTIONS, &Vec::new())
                .await?;
        }

        // Extract Vector Store ID
        let vector_store_id = if let Some(id) = &vector_store.id {
            id.to_string()
        } else {
            return Err(anyhow!(
                "[OpenAI][Assistants] Unable to attach Vector Store. No valid ID found."
            ));
        };

        //Get version-specific URL
        let assistant_resource = OpenAIAssistantResource::Assistant {
            assistant_id: self.id.clone().unwrap_or_default(),
        };
        let assistant_url = self.version.get_endpoint(&assistant_resource);

        //Get version-specific headers
        let version_headers = self.version.get_headers(&self.api_key);

        let body = json!({
            "tool_resources": {
                "file_search": {
                    "vector_store_ids": vec![vector_store_id]
                }
            },
        });

        //Make the API call
        let client = Client::new();

        let response = client
            .post(assistant_url)
            .headers(version_headers)
            .json(&body)
            .send()
            .await?;

        let response_status = response.status();
        let response_text = response.text().await?;

        if self.debug {
            info!(
                "[debug] OpenAI Vector Store Attach API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        //Deserialize the string response into the Assistants object to confirm if there were any errors
        serde_json::from_str::<OpenAIAssistantResp>(&response_text)
            .map_err(|error| {
                let error = AllmsError {
                    crate_name: "allms".to_string(),
                    module: "assistants::openai_assistant".to_string(),
                    error_message: format!(
                        "Vector Store Attach API response serialization error: {}",
                        error
                    ),
                    error_detail: response_text,
                };
                error!("{:?}", error);
                anyhow!("{:?}", error)
            })
            .map(|_| Ok(()))?
    }
}
