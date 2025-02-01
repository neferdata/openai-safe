use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::info;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    constants::{OPENAI_API_URL, OPENAI_BASE_INSTRUCTIONS, OPENAI_FUNCTION_INSTRUCTIONS},
    domain::{OpenAPIChatResponse, OpenAPICompletionsResponse, RateLimit},
    llm_models::LLMModel,
    utils::map_to_range,
};

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq)]
pub enum OpenAIModels {
    Gpt3_5Turbo,
    Gpt3_5Turbo0613,
    Gpt3_5Turbo16k,
    Gpt4,
    Gpt4_32k,
    TextDavinci003,
    Gpt4Turbo,
    Gpt4TurboPreview,
    Gpt4o,
    Gpt4o20240806,
    Gpt4oMini,
    // Reasoning models
    O1Preview,
    O1Mini,
    O1,
    O3Mini,
    // Custom models
    Custom { name: String },
}

#[async_trait(?Send)]
impl LLMModel for OpenAIModels {
    fn as_str(&self) -> &str {
        match self {
            OpenAIModels::Gpt3_5Turbo => "gpt-3.5-turbo",
            OpenAIModels::Gpt3_5Turbo0613 => "gpt-3.5-turbo-0613",
            OpenAIModels::Gpt3_5Turbo16k => "gpt-3.5-turbo-16k",
            OpenAIModels::Gpt4 => "gpt-4",
            OpenAIModels::Gpt4_32k => "gpt-4-32k",
            OpenAIModels::TextDavinci003 => "text-davinci-003",
            OpenAIModels::Gpt4Turbo => "gpt-4-turbo",
            OpenAIModels::Gpt4TurboPreview => "gpt-4-turbo-preview",
            OpenAIModels::Gpt4o => "gpt-4o",
            OpenAIModels::Gpt4o20240806 => "gpt-4o-2024-08-06",
            OpenAIModels::Gpt4oMini => "gpt-4o-mini",
            OpenAIModels::O1Preview => "o1-preview",
            OpenAIModels::O1Mini => "o1-mini",
            OpenAIModels::O1 => "o1",
            OpenAIModels::O3Mini => "o3-mini",
            OpenAIModels::Custom { name } => name.as_str(),
        }
    }

    fn try_from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "gpt-3.5-turbo" => Some(OpenAIModels::Gpt3_5Turbo),
            "gpt-3.5-turbo-0613" => Some(OpenAIModels::Gpt3_5Turbo0613),
            "gpt-3.5-turbo-16k" => Some(OpenAIModels::Gpt3_5Turbo16k),
            "gpt-4" => Some(OpenAIModels::Gpt4),
            "gpt-4-32k" => Some(OpenAIModels::Gpt4_32k),
            "text-davinci-003" => Some(OpenAIModels::TextDavinci003),
            "gpt-4-turbo" => Some(OpenAIModels::Gpt4Turbo),
            "gpt-4-turbo-preview" => Some(OpenAIModels::Gpt4TurboPreview),
            "gpt-4o" => Some(OpenAIModels::Gpt4o),
            "gpt-4o-2024-08-06" => Some(OpenAIModels::Gpt4o20240806),
            "gpt-4o-mini" => Some(OpenAIModels::Gpt4oMini),
            "o1-preview" => Some(OpenAIModels::O1Preview),
            "o1-mini" => Some(OpenAIModels::O1Mini),
            "o1" => Some(OpenAIModels::O1),
            "o3-mini" => Some(OpenAIModels::O3Mini),
            _ => Some(OpenAIModels::Custom {
                name: name.to_string(),
            }),
        }
    }

    fn default_max_tokens(&self) -> usize {
        //OpenAI documentation: https://platform.openai.com/docs/models/gpt-3-5
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt3_5Turbo => 4096,
            OpenAIModels::Gpt3_5Turbo0613 => 4096,
            OpenAIModels::Gpt3_5Turbo16k => 16384,
            OpenAIModels::Gpt4 => 8192,
            OpenAIModels::Gpt4_32k => 32768,
            OpenAIModels::TextDavinci003 => 4097,
            OpenAIModels::Gpt4Turbo => 128_000,
            OpenAIModels::Gpt4TurboPreview => 128_000,
            OpenAIModels::Gpt4o => 128_000,
            OpenAIModels::Gpt4o20240806 => 128_000,
            OpenAIModels::Gpt4oMini => 128_000,
            OpenAIModels::O1Preview => 128_000,
            OpenAIModels::O1Mini => 128_000,
            OpenAIModels::O1 => 200_000,
            OpenAIModels::O3Mini => 200_000,
            OpenAIModels::Custom { .. } => 128_000,
        }
    }

    fn get_endpoint(&self) -> String {
        //OpenAI documentation: https://platform.openai.com/docs/models/model-endpoint-compatibility
        match self {
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4TurboPreview
            | OpenAIModels::Gpt4o
            | OpenAIModels::Gpt4o20240806
            | OpenAIModels::Gpt4oMini
            | OpenAIModels::Gpt4_32k
            | OpenAIModels::O1Preview
            | OpenAIModels::O1Mini
            | OpenAIModels::O1
            | OpenAIModels::O3Mini
            | OpenAIModels::Custom { .. } => {
                format!(
                    "{OPENAI_API_URL}/v1/chat/completions",
                    OPENAI_API_URL = *OPENAI_API_URL
                )
            }
            OpenAIModels::TextDavinci003 => format!(
                "{OPENAI_API_URL}/v1/completions",
                OPENAI_API_URL = *OPENAI_API_URL
            ),
        }
    }

    fn get_base_instructions(&self, function_call: Option<bool>) -> String {
        let function_call = function_call.unwrap_or_else(|| self.function_call_default());
        match function_call {
            true => OPENAI_FUNCTION_INSTRUCTIONS.to_string(),
            false => OPENAI_BASE_INSTRUCTIONS.to_string(),
        }
    }

    fn function_call_default(&self) -> bool {
        //OpenAI documentation: https://platform.openai.com/docs/guides/gpt/function-calling
        match self {
            OpenAIModels::TextDavinci003
            | OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt4_32k
            | OpenAIModels::O1Preview
            | OpenAIModels::O1
            | OpenAIModels::O1Mini
            | OpenAIModels::O3Mini => false,
            OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4TurboPreview
            | OpenAIModels::Gpt4o
            | OpenAIModels::Gpt4o20240806
            | OpenAIModels::Gpt4oMini
            | OpenAIModels::Custom { .. } => true,
        }
    }

    //This method prepares the body of the API call for different models
    fn get_body(
        &self,
        instructions: &str,
        json_schema: &Value,
        function_call: bool,
        max_tokens: &usize,
        temperature: &f32,
    ) -> serde_json::Value {
        match self {
            //https://platform.openai.com/docs/api-reference/completions/create
            //For DaVinci model all text goes into the 'prompt' filed of the body
            OpenAIModels::TextDavinci003 => {
                let schema_string = serde_json::to_string(json_schema).unwrap_or_default();
                let base_instructions = self.get_base_instructions(Some(function_call));
                json!({
                    "model": self.as_str(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt": format!(
                        "{base_instructions}\n\n
                        Output Json schema:\n
                        {schema_string}\n\n
                        {instructions}",
                    ),
                })
            }
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4TurboPreview
            | OpenAIModels::Gpt4o
            | OpenAIModels::Gpt4o20240806
            | OpenAIModels::Gpt4oMini
            | OpenAIModels::Gpt4_32k
            | OpenAIModels::Custom { .. } => {
                let base_instructions = self.get_base_instructions(Some(function_call));
                let system_message = json!({
                    "role": "system",
                    "content": base_instructions,
                });

                match function_call {
                    //If we choose to use function calling
                    //https://platform.openai.com/docs/guides/gpt/function-calling
                    true => {
                        let user_message = json!({
                            "role": "user",
                            "content": instructions,
                        });

                        let function = json!({
                            "name": "analyze_data",
                            "description": "Use this function to compute the answer based on input data, instructions and your language model. Output should be a fully formed JSON object.",
                            "parameters": json_schema,
                        });

                        let function_call = json!({
                            "name": "analyze_data"
                        });

                        //For ChatGPT we ignore max_tokens. It will default to 'inf'
                        json!({
                            "model": self.as_str(),
                            "temperature": temperature,
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                            "functions": vec![
                                function,
                            ],
                            //This forces ChatGPT to use the function definition
                            "function_call": function_call,
                        })
                    }
                    //https://platform.openai.com/docs/guides/chat/introduction
                    false => {
                        let schema_string = serde_json::to_string(json_schema).unwrap_or_default();

                        let user_message = json!({
                            "role": "user",
                            "content": format!(
                                "Output Json schema:\n
                                {schema_string}\n\n
                                {instructions}"
                            ),
                        });
                        //For ChatGPT we ignore max_tokens. It will default to 'inf'
                        json!({
                            "model": self.as_str(),
                            "temperature": temperature,
                            "messages": vec![
                                system_message,
                                user_message,
                            ],
                        })
                    }
                }
            }
            // Review https://platform.openai.com/docs/guides/reasoning for beta limitations:
            // - Message types: user and assistant messages only, system messages are not supported.
            // - Tools: tools, function calling, and response format parameters are not supported.
            // - Other: temperature, top_p and n are fixed at 1, while presence_penalty and frequency_penalty are fixed at 0.
            // - Assistants and Batch: these models are not supported in the Assistants API or Batch API.
            OpenAIModels::O1Preview
            | OpenAIModels::O1Mini
            | OpenAIModels::O1
            | OpenAIModels::O3Mini => {
                let base_instructions = self.get_base_instructions(Some(function_call));
                let system_message = json!({
                    "role": "user",
                    "content": base_instructions,
                });

                let schema_string = serde_json::to_string(json_schema).unwrap_or_default();

                let user_message = json!({
                    "role": "user",
                    "content": format!(
                        "Output Json schema:\n
                        {schema_string}\n\n
                        {instructions}"
                    ),
                });
                json!({
                    "model": self.as_str(),
                    "messages": vec![
                        system_message,
                        user_message,
                    ],
                })
            }
        }
    }
    /*
     * This function leverages OpenAI API to perform any query as per the provided body.
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

        let response_status = response.status();
        let response_text = response.text().await?;

        if debug {
            info!(
                "[debug] OpenAI API response: [{}] {:#?}",
                &response_status, &response_text
            );
        }

        Ok(response_text)
    }

    //This method attempts to convert the provided API response text into the expected struct and extracts the data from the response
    fn get_data(&self, response_text: &str, function_call: bool) -> Result<String> {
        match self {
            //https://platform.openai.com/docs/api-reference/completions/create
            OpenAIModels::TextDavinci003 => {
                //Convert API response to struct representing expected response format
                let completions_response: OpenAPICompletionsResponse =
                    serde_json::from_str(response_text)?;

                //Extract data part
                match completions_response.choices {
                    Some(choices) => Ok(choices.into_iter().filter_map(|item| item.text).collect()),
                    None => Err(anyhow!(
                        "Unable to retrieve response from OpenAI Completions API"
                    )),
                }
            }
            //https://platform.openai.com/docs/guides/chat/introduction
            OpenAIModels::Gpt3_5Turbo
            | OpenAIModels::Gpt3_5Turbo0613
            | OpenAIModels::Gpt3_5Turbo16k
            | OpenAIModels::Gpt4
            | OpenAIModels::Gpt4Turbo
            | OpenAIModels::Gpt4TurboPreview
            | OpenAIModels::Gpt4o
            | OpenAIModels::Gpt4o20240806
            | OpenAIModels::Gpt4oMini
            | OpenAIModels::Gpt4_32k
            | OpenAIModels::O1Preview
            | OpenAIModels::O1Mini
            | OpenAIModels::O1
            | OpenAIModels::O3Mini
            | OpenAIModels::Custom { .. } => {
                //Convert API response to struct representing expected response format
                let chat_response: OpenAPIChatResponse = serde_json::from_str(response_text)?;

                //Extract data part
                match chat_response.choices {
                    Some(choices) => Ok(choices
                        .into_iter()
                        .filter_map(|item| {
                            //For function_call the response is in arguments, and for regular call in content
                            match function_call {
                                true => item.message.function_call.map(|function_call| {
                                    self.sanitize_json_response(&function_call.arguments)
                                }),
                                false => item
                                    .message
                                    .content
                                    .map(|content| self.sanitize_json_response(&content)),
                            }
                        })
                        .collect()),
                    None => Err(anyhow!("Unable to retrieve response from OpenAI Chat API")),
                }
            }
        }
    }

    /// This function allows to check the rate limits for different models
    /// Rate limit for `Custom` model is assumed based on `GPT-4o` limits
    fn get_rate_limit(&self) -> RateLimit {
        //OpenAI documentation: https://platform.openai.com/account/rate-limits
        //This is the max tokens allowed between prompt & response
        match self {
            OpenAIModels::Gpt3_5Turbo => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo0613 => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt3_5Turbo16k => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4 => RateLimit {
                tpm: 300_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4Turbo => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4TurboPreview => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4_32k => RateLimit {
                tpm: 300_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4o | OpenAIModels::Custom { .. } => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4o20240806 => RateLimit {
                tpm: 2_000_000,
                rpm: 10_000,
            },
            OpenAIModels::Gpt4oMini => RateLimit {
                tpm: 1_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O1Preview => RateLimit {
                tpm: 30_000_000,
                rpm: 10_000,
            },
            OpenAIModels::O1Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::O1 => RateLimit {
                tpm: 30_000_000,
                rpm: 1_000,
            },
            OpenAIModels::O3Mini => RateLimit {
                tpm: 150_000_000,
                rpm: 30_000,
            },
            OpenAIModels::TextDavinci003 => RateLimit {
                tpm: 250_000,
                rpm: 3_000,
            },
        }
    }

    // Accepts a [0-100] percentage range and returns the target temperature based on model ranges
    fn get_normalized_temperature(&self, relative_temp: u32) -> f32 {
        // Temperature range documentation: https://platform.openai.com/docs/api-reference/chat/create
        let min = 0u32;
        let max = 2u32;
        map_to_range(min, max, relative_temp)
    }
}

impl OpenAIModels {
    // This function checks if a model supports tool use in Assistants API (e.g. file_search)
    pub fn tools_support(&self) -> bool {
        matches!(
            self,
            OpenAIModels::Gpt3_5Turbo
                | OpenAIModels::Gpt4Turbo
                | OpenAIModels::Gpt4TurboPreview
                | OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Custom { .. }
        )
    }

    // This function checks if a model supports Structured Outputs
    // https://openai.com/index/introducing-structured-outputs-in-the-api/
    pub fn structured_output_support(&self) -> bool {
        matches!(
            self,
            OpenAIModels::Gpt4o
                | OpenAIModels::Gpt4o20240806
                | OpenAIModels::Gpt4oMini
                | OpenAIModels::Custom { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::llm_models::llm_model::LLMModel;
    use crate::llm_models::OpenAIModels;

    // Tests for calculating max requests per model
    #[test]
    fn test_gpt3_5turbo_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10000, 2000000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo0613_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo0613;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10000, 2000000 / ((4096_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt3_5turbo16k_max_requests() {
        let model = OpenAIModels::Gpt3_5Turbo16k;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10000, 2000000 / ((16384_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    #[test]
    fn test_gpt4_max_requests() {
        let model = OpenAIModels::Gpt4;
        let max_requests = model.get_max_requests();
        let expected_max = std::cmp::min(10_000, 300_000 / ((8192_f64 * 0.5).ceil() as usize));
        assert_eq!(max_requests, expected_max);
    }

    // Tests of model creation
    #[test]
    fn test_try_from_str_standard_models() {
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo"),
            Some(OpenAIModels::Gpt3_5Turbo)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo-0613"),
            Some(OpenAIModels::Gpt3_5Turbo0613)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-3.5-turbo-16k"),
            Some(OpenAIModels::Gpt3_5Turbo16k)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4"),
            Some(OpenAIModels::Gpt4)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-32k"),
            Some(OpenAIModels::Gpt4_32k)
        );
        assert_eq!(
            OpenAIModels::try_from_str("text-davinci-003"),
            Some(OpenAIModels::TextDavinci003)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-turbo"),
            Some(OpenAIModels::Gpt4Turbo)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4-turbo-preview"),
            Some(OpenAIModels::Gpt4TurboPreview)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o"),
            Some(OpenAIModels::Gpt4o)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o-2024-08-06"),
            Some(OpenAIModels::Gpt4o20240806)
        );
        assert_eq!(
            OpenAIModels::try_from_str("gpt-4o-mini"),
            Some(OpenAIModels::Gpt4oMini)
        );
    }

    #[test]
    fn test_try_from_str_case_insensitivity() {
        assert_eq!(
            OpenAIModels::try_from_str("GPT-4"),
            Some(OpenAIModels::Gpt4)
        );
        assert_eq!(
            OpenAIModels::try_from_str("GPT-4o-MiNI"),
            Some(OpenAIModels::Gpt4oMini)
        );
    }

    #[test]
    fn test_try_from_str_custom_model() {
        assert_eq!(
            OpenAIModels::try_from_str("my-custom-model"),
            Some(OpenAIModels::Custom {
                name: "my-custom-model".to_string()
            })
        );
        assert_eq!(
            OpenAIModels::try_from_str("AnotherModel"),
            Some(OpenAIModels::Custom {
                name: "AnotherModel".to_string()
            })
        );
    }
}
