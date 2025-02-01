use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

use allms::{
    llm::{
        AnthropicModels, AwsBedrockModels, DeepSeekModels, GoogleModels, LLMModel, MistralModels,
        OpenAIModels, PerplexityModels,
    },
    Completions,
};

#[derive(Deserialize, Serialize, JsonSchema, Debug, Clone)]
struct TranslationResponse {
    pub spanish: String,
    pub french: String,
    pub german: String,
    pub polish: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // Example context and instructions
    let instructions =
        "Translate the following English sentence to all the languages in the response type: Rust is best for working with LLMs";

    // Get answer using AWS Bedrock Converse
    // AWS Bedrock SDK requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to be defined and matching your AWS account
    let model = AwsBedrockModels::try_from_str("amazon.nova-lite-v1:0")
        .unwrap_or(AwsBedrockModels::NovaLite); // Choose the model
    println!("AWS Bedrock model: {:#?}", model.as_str());

    let aws_completion = Completions::new(model, "", None, None);

    match aws_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("AWS Bedrock response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using OpenAI
    let openai_api_key: String = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = OpenAIModels::try_from_str("o3-mini").unwrap_or(OpenAIModels::Gpt4oMini); // Choose the model
    println!("OpenAI model: {:#?}", model.as_str());

    let openai_completion = Completions::new(model, &openai_api_key, None, None);

    match openai_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("OpenAI response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Anthropic
    let anthropic_api_key: String =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = AnthropicModels::try_from_str("claude-3-5-sonnet-20240620")
        .unwrap_or(AnthropicModels::Claude3_5Sonnet); // Choose the model
    println!("Anthropic model: {:#?}", model.as_str());

    let anthropic_completion = Completions::new(model, &anthropic_api_key, None, None);

    match anthropic_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Anthropic response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Mistral
    let mistral_api_key: String =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
    let model =
        MistralModels::try_from_str("open-mistral-nemo").unwrap_or(MistralModels::MistralLarge); // Choose the model
    println!("Mistral model: {:#?}", model.as_str());

    let mistral_completion = Completions::new(model, &mistral_api_key, None, None);

    match mistral_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Mistral response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Google GeminiPro
    let model =
        GoogleModels::try_from_str("gemini-1.5-pro").unwrap_or(GoogleModels::Gemini1_5Flash); // Choose the model
    println!("Google Gemini model: {:#?}", model.as_str());

    let google_token_str: String =
        std::env::var("GOOGLE_AI_STUDIO_API_KEY").expect("GOOGLE_AI_STUDIO_API_KEY not set");

    let gemini_completion = Completions::new(model, &google_token_str, None, None);

    match gemini_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Gemini response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using Perplexity
    let model =
        PerplexityModels::try_from_str("sonar-reasoning").unwrap_or(PerplexityModels::Sonar); // Choose the model
    println!("Perplexity model: {:#?}", model.as_str());

    let perplexity_token_str: String =
        std::env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set");

    let perplexity_completion = Completions::new(model, &perplexity_token_str, None, None);

    match perplexity_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("Perplexity response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    // Get answer using DeepSeek
    let model =
        DeepSeekModels::try_from_str("deepseek-reasoner").unwrap_or(DeepSeekModels::DeepSeekChat); // Choose the model
    println!("DeepSeek model: {:#?}", model.as_str());

    let deepseek_token_str: String =
        std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set");

    let deepseek_completion = Completions::new(model, &deepseek_token_str, None, None);

    match deepseek_completion
        .get_answer::<TranslationResponse>(instructions)
        .await
    {
        Ok(response) => println!("DeepSeek response: {:#?}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
