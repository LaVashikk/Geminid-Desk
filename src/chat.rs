#[cfg(feature = "tts")]
use crate::sessions::SharedTts;

use crate::{
    easymark::MemoizedEasymarkHighlighter,
    file_handler::{Attachment, AttachmentState},
    widgets::{self, GeminiModel, ModelPicker, Settings},
};
use anyhow::{Context, Result};
use eframe::egui::{
    self, Align, Color32, CornerRadius, Frame, Id, Key, KeyboardShortcut, Layout, Margin, Modifiers, Pos2, Rect, Stroke, TextStyle, pos2, vec2
};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use egui_modal::{Icon, Modal};
use egui_robust_scroll::RobustVirtualScroll;
use flowync::{error::Compact, CompactFlower, CompactHandle};
use futures_util::TryStreamExt;
use gemini_rust::{
    Gemini, GenerationConfig, HarmBlockThreshold, HarmCategory, Part, SafetySetting, UsageMetadata,
};
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio_stream::StreamExt;

const SAFETY_SETTINGS: [SafetySetting; 4] = [
    SafetySetting {
        category: HarmCategory::Harassment,
        threshold: HarmBlockThreshold::BlockNone,
    },
    SafetySetting {
        category: HarmCategory::HateSpeech,
        threshold: HarmBlockThreshold::BlockNone,
    },
    SafetySetting {
        category: HarmCategory::SexuallyExplicit,
        threshold: HarmBlockThreshold::BlockNone,
    },
    SafetySetting {
        category: HarmCategory::DangerousContent,
        threshold: HarmBlockThreshold::BlockNone,
    },
];

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Message {
    pub model: GeminiModel,
    pub content: String,
    pub role: MessageRole,
    #[serde(skip)]
    pub is_generating: bool,
    #[serde(skip)]
    pub requested_at: Instant,
    pub time: chrono::DateTime<chrono::Utc>,
    pub generation_time: Option<Duration>,
    #[serde(skip)]
    pub clicked_copy: bool,
    pub is_error: bool,
    #[serde(skip)]
    pub is_speaking: bool,
    pub files: Vec<Attachment>,
    pub is_prepending: bool,
    pub is_thought: bool,
    pub usage: Option<UsageMetadata>,
    #[serde(skip)]
    pub status_message: Option<String>,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            content: String::new(),
            role: MessageRole::User,
            is_generating: false,
            requested_at: Instant::now(),
            time: chrono::Utc::now(),
            clicked_copy: false,
            is_error: false,
            is_speaking: false,
            model: GeminiModel::default(),
            files: Vec::new(),
            is_prepending: false,
            is_thought: false,
            generation_time: None,
            usage: None,
            status_message: None,
        }
    }
}

#[cfg(feature = "tts")]
fn tts_control(tts: SharedTts, text: String, speak: bool) {
    std::thread::spawn(move || {
        if let Some(tts) = tts {
            if speak {
                let _ = tts
                    .write()
                    .speak(widgets::sanitize_text_for_tts(&text), true)
                    .map_err(|e| log::error!("failed to speak: {e}"));
            } else {
                let _ = tts
                    .write()
                    .stop()
                    .map_err(|e| log::error!("failed to stop tts: {e}"));
            }
        }
    });
}

fn make_short_name(_name: &str) -> String {
    // todo stuff
    // let mut c = name
    //     .split('/')
    //     .next()
    //     .unwrap_or(name)
    //     .chars()
    //     .take_while(|c| c.is_alphanumeric());
    // match c.next() {
    //     None => "Gemini".to_string(),
    //     Some(f) => f.to_uppercase().collect::<String>() + c.collect::<String>().as_str(),
    // }
    "Gemini".to_string()
}

enum MessageAction {
    None,
    Retry(usize),
    Regenerate(usize),
    Delete(usize),
}

impl Message {
    #[inline]
    fn user(content: String, model: GeminiModel, files: Vec<Attachment>) -> Self {
        Self {
            content,
            role: MessageRole::User,
            is_generating: false,
            model,
            files,
            ..Default::default()
        }
    }

    #[inline]
    fn assistant(content: String, model: GeminiModel) -> Self {
        Self {
            content,
            role: MessageRole::Assistant,
            is_generating: true,
            model,
            ..Default::default()
        }
    }

    #[inline]
    const fn is_user(&self) -> bool {
        matches!(self.role, MessageRole::User)
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        commonmark_cache: &mut CommonMarkCache,
        #[cfg(feature = "tts")] tts: SharedTts,
        idx: usize,
        prepend_buf: &mut String,
    ) -> MessageAction {
        // message role
        let message_offset = ui
            .horizontal(|ui| {
                if self.is_user() {
                    let f = ui.label("👤").rect.left();
                    ui.label("You").rect.left() - f
                } else {
                    let f = ui.label("✨").rect.left();
                    let offset = ui
                        .label(make_short_name(&self.model.to_string()))
                        .on_hover_text(&self.model.to_string())
                        .rect
                        .left()
                        - f;
                    // ui.add_enabled(false, egui::Label::new(&self.model.to_string())); //? todo redundant?
                    if let Some(duration) = self.generation_time {
                        ui.weak(format!("({:.1}s)", duration.as_secs_f64()))
                            .on_hover_text("Generation time");
                    }
                    if let Some(usage) = &self.usage {
                        let total = usage.total_token_count.unwrap_or(0);
                        let text = format!(
                            "In: {} / Out: {} / Total: {}",
                            usage.prompt_token_count.unwrap_or(0),
                            usage.candidates_token_count.unwrap_or(0),
                            total
                        );
                        ui.weak(format!("{} ᵗ", total)).on_hover_text(text);
                    }
                    offset
                }
            })
            .inner;

        let is_commonmark = !self.content.is_empty() && !self.is_error && !self.is_prepending;
        if is_commonmark && !self.is_thought {
            ui.add_space(-TextStyle::Body.resolve(ui.style()).size + 4.0);
        }

        // message content / spinner
        let mut action = MessageAction::None;
        ui.horizontal(|ui| {
            ui.add_space(message_offset);
            if self.content.is_empty() && self.is_generating && !self.is_error {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());

                    if let Some(status) = &self.status_message {
                        ui.label(status);
                    } else {
                        // show time spent waiting for response
                        ui.add_enabled(
                            false,
                            egui::Label::new(format!(
                                "{:.1}s",
                                self.requested_at.elapsed().as_secs_f64()
                            )),
                        );
                    }
                });
            } else if self.is_error {
                ui.vertical(|ui| {
                    CommonMarkViewer::new().show(ui, commonmark_cache, &self.content);
                    ui.add_space(8.0);
                    if ui
                        .button("🔄 Retry Generation")
                        .on_hover_text(
                            "Try to generate a response again. Make sure you have a valid API Key and stable connection.",
                        )
                        .clicked()
                    {
                        action = MessageAction::Retry(idx);
                    }
                });
            } else if self.is_prepending {
                let textedit = ui.add(
                    egui::TextEdit::multiline(prepend_buf).hint_text("Prepend text to response…"),
                );
                macro_rules! cancel_prepend {
                    () => {
                        self.is_prepending = false;
                        prepend_buf.clear();
                    };
                }
                if textedit.lost_focus() && ui.input(|i| i.key_pressed(Key::Escape)) {
                    cancel_prepend!();
                }
                ui.vertical(|ui| {
                    if ui
                        .button("🔄 Regenerate")
                        .on_hover_text(
                            "Generate the response again, \
                            the LLM will start after any prepended text",
                        )
                        .clicked()
                    {
                        self.content = prepend_buf.clone();
                        self.is_prepending = false;
                        self.is_generating = true;
                        action = MessageAction::Regenerate(idx);
                    }
                    if !prepend_buf.is_empty()
                        && ui
                            .button("\u{270f} Edit")
                            .on_hover_text(
                                "Edit the message in the context, but don't regenerate it",
                            )
                            .clicked()
                    {
                        self.content = prepend_buf.clone();
                        cancel_prepend!();
                    }
                    if ui.button("❌ Cancel").clicked() {
                        cancel_prepend!();
                    }
                });
            } else {
                if self.is_thought {
                    ui.horizontal(|ui| {
                        let done_thinking = !self.is_generating;
                        Frame::group(ui.style())
                            .inner_margin(Margin::symmetric(8, 4))
                            .show(ui, |ui| {
                                // egui::collapsing_header::CollapsingState::load_with_default_open
                                egui::CollapsingHeader::new("  Thoughts")
                                    .id_salt(self.time.timestamp_millis())
                                    .default_open(false)
                                    .icon(move |ui, openness, response| {
                                        widgets::thinking_icon(
                                            ui,
                                            openness,
                                            response,
                                            done_thinking,
                                        );
                                    })
                                    .show(ui, |ui| {
                                        CommonMarkViewer::new().show(
                                            ui,
                                            commonmark_cache,
                                            &self.content,
                                        );
                                    });
                            });
                    });
                    ui.add_space(4.0);
                } else {
                    CommonMarkViewer::new().max_image_width(Some(512)).show(
                        ui,
                        commonmark_cache,
                        &self.content,
                    );
                }
            }
        });

        // files
        if !self.files.is_empty() {
            if is_commonmark {
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                ui.add_space(message_offset);
                egui::ScrollArea::horizontal().id_salt(idx).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        crate::file_handler::show_files(ui, &mut self.files, false);
                    });
                })
            });
            ui.add_space(8.0);
        }

        if self.is_prepending {
            return action;
        }

        // copy buttons and such
        // let shift_held = !ui.ctx().wants_keyboard_input() && ui.input(|i| i.modifiers.shift);

        if !self.is_generating && !self.is_error {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_space(message_offset);
                if !self.content.is_empty() {
                    let copy = ui
                        .add(
                            egui::Button::new(if self.clicked_copy { "✔" } else { "🗐" })
                                .small()
                                .fill(egui::Color32::TRANSPARENT),
                        )
                        .on_hover_text(if self.clicked_copy {
                            "Copied!"
                        } else {
                            "Copy message"
                        });
                    if copy.clicked() {
                        ui.ctx().copy_text(self.content.clone());
                        self.clicked_copy = true;
                    }
                    self.clicked_copy = self.clicked_copy && copy.hovered();
                }

                #[cfg(feature = "tts")]
                {
                    let speak = ui
                        .add(
                            egui::Button::new(if self.is_speaking { "…" } else { "🔊" })
                                .small()
                                .fill(egui::Color32::TRANSPARENT),
                        )
                        .on_hover_text("Read the message out loud. Right click to repeat");

                    if speak.clicked() {
                        if self.is_speaking {
                            self.is_speaking = false;
                            tts_control(tts, String::new(), false);
                        } else {
                            self.is_speaking = true;
                            tts_control(tts, self.content.clone(), true);
                        }
                    } else if speak.secondary_clicked() {
                        self.is_speaking = true;
                        tts_control(tts, self.content.clone(), true);
                    }
                }

                if ui
                    .add(
                        egui::Button::new("🗑")
                            .small()
                            .fill(egui::Color32::TRANSPARENT),
                    )
                    .on_hover_text("Remove")
                    .clicked()
                {
                    action = MessageAction::Delete(idx);
                }

                if !self.is_user()
                    && !self.is_thought
                    && prepend_buf.is_empty()
                    && ui
                        .add(
                            egui::Button::new("🔄")
                                .small()
                                .fill(egui::Color32::TRANSPARENT),
                        )
                        .on_hover_text("Regenerate")
                        .clicked()
                {
                    prepend_buf.clear();
                    self.is_prepending = true;
                }
            });
        }
        ui.add_space(12.0);

        action
    }
}

// <completion progress, final completion, error>
#[derive(Debug, Clone)]
pub enum ChatProgress {
    Part(Part),
    // Status update (e.g. "Uploading files...")
    Status {
        message: String,
    },
    FileUploading {
        path: PathBuf,
    },
    FileUploaded {
        path: PathBuf,
        file: gemini_rust::File,
    },
}

pub type CompletionFlower =
    CompactFlower<(usize, ChatProgress), (usize, String, Option<UsageMetadata>), (usize, String)>;
pub type CompletionFlowerHandle =
    CompactHandle<(usize, ChatProgress), (usize, String, Option<UsageMetadata>), (usize, String)>;

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Chat {
    pub chatbox: String,
    pub messages: Vec<Message>,
    pub summary: String,
    pub stop_generating: Arc<AtomicBool>,
    pub model_picker: ModelPicker,
    pub files: Vec<Attachment>,
    pub prepend_buf: String,

    #[serde(skip)]
    pub token_count: Option<u32>,
    #[serde(skip)]
    pub last_content_hash: u64,
    #[serde(skip)]
    pub last_token_check: Option<Instant>,

    #[serde(skip)]
    pub chatbox_height: f32,
    #[serde(skip)]
    pub flower: CompletionFlower,
    #[serde(skip)]
    pub retry_message_idx: Option<usize>,
    #[serde(skip)]
    pub chatbox_highlighter: MemoizedEasymarkHighlighter,
}

impl Default for Chat {
    fn default() -> Self {
        Self {
            chatbox: String::new(),
            chatbox_height: 0.0,
            messages: Vec::new(),
            flower: CompletionFlower::new(1),
            retry_message_idx: None,
            summary: String::new(),
            chatbox_highlighter: MemoizedEasymarkHighlighter::default(),
            stop_generating: Arc::new(AtomicBool::new(false)),
            model_picker: ModelPicker::default(),
            files: Vec::new(),
            prepend_buf: String::new(),
            token_count: None,
            last_content_hash: 0,
            last_token_check: None,
        }
    }
}

async fn request_completion(
    gemini: Gemini,
    messages: Vec<Message>,
    handle: &CompletionFlowerHandle,
    stop_generating: Arc<AtomicBool>,
    index: usize,
    use_streaming: bool,
    public_file_upload: bool,
    generation_config: GenerationConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Requesting completion... (history length: {})",
        messages.len()
    );

    let history = crate::chat_completion::build_history(
        &gemini,
        &messages,
        None,
        public_file_upload,
        Some((index, handle)),
    )
    .await?;

    // 2. Prepare the request builder
    let mut content_builder = gemini.generate_content();

    // Inject constructed history
    content_builder.contents.extend(history);

    // Apply configuration
    let content_builder_final = content_builder
        .with_safety_settings(SAFETY_SETTINGS.to_vec())
        .with_generation_config(generation_config);

    let mut response_text = String::new();
    let mut final_usage = None;

    // Helper closure for cancellation polling
    let check_cancellation = || async {
        loop {
            if stop_generating.load(Ordering::SeqCst) {
                stop_generating.store(false, Ordering::SeqCst);
                log::warn!("Request cancelled");
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };

    // 3. Execute Request (Streaming or Blocking)
    if use_streaming {
        // Start the stream, respecting cancellation immediately
        let stream_result = tokio::select! {
            _ = check_cancellation() => None,
            res = content_builder_final.execute_stream() => Some(res),
        };

        let mut stream = match stream_result {
            Some(Ok(s)) => s.into_stream(),
            Some(Err(e)) => return Err(e.into()),
            None => {
                handle.success((index, response_text, final_usage));
                return Ok(());
            }
        };

        log::info!("Reading stream response...");

        // Consume the stream
        loop {
            tokio::select! {
                _ = check_cancellation() => {
                    log::info!("Streaming generation cancelled by user.");
                    break;
                }
                next_item = stream.next() => {
                    match next_item {
                        Some(Ok(res)) => {
                            // Capture usage metadata if available
                            if let Some(usage) = res.usage_metadata {
                                final_usage = Some(usage);
                            }

                            // Process candidates
                            if let Some(candidate) = res.candidates.first() {
                                if let Some(parts) = &candidate.content.parts {
                                    for part in parts {
                                        // Send intermediate part to UI
                                        handle.send((index, ChatProgress::Part(part.clone())));

                                        // Accumulate full text for final state
                                        if let Part::Text { text, .. } = part {
                                            response_text += &text;
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => return Err(e.into()),
                        None => break, // Stream exhausted
                    }
                }
            }
        }
    } else {
        log::info!("Sending non-streaming request...");

        tokio::select! {
            _ = check_cancellation() => {}
            result = content_builder_final.execute() => {
                match result {
                    Ok(response) => {
                        log::info!("Non-streaming response received.");
                        final_usage = response.usage_metadata;

                        if let Some(candidate) = response.candidates.first() {
                            if let Some(parts) = &candidate.content.parts {
                                for part in parts {
                                    handle.send((index, ChatProgress::Part(part.clone())));
                                    if let Part::Text { text, .. } = part {
                                        response_text += text;
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => return Err(Box::new(err)),
                }
            }
        }
    }

    log::info!(
        "Completion request finished. Total response length: {}",
        response_text.len()
    );

    // Notify UI of success
    handle.success((index, response_text, final_usage));

    Ok(())
}

async fn request_completion_code_assist(
    client: gemini_code_assist_adapter::CodeAssistClient,
    messages: Vec<Message>,
    handle: &CompletionFlowerHandle,
    stop_generating: Arc<AtomicBool>,
    index: usize,
    use_streaming: bool,
    generation_config: GenerationConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::info!(
        "Requesting completion via Code Assist... (history length: {})",
        messages.len()
    );

    let dummy_client = Gemini::new("")?;

    let history = crate::chat_completion::build_history(
        &dummy_client,
        &messages,
        None,
        false,
        Some((index, handle)),
    )
    .await?;

    let gemini_request = gemini_rust::GenerateContentRequest {
        contents: history,
        generation_config: Some(generation_config),
        safety_settings: Some(SAFETY_SETTINGS.to_vec()),
        tools: None,
        tool_config: None,
        system_instruction: None,
        cached_content: None,
    };

    let mut response_text = String::new();
    let mut final_usage = None;

    let check_cancellation = || async {
        loop {
            if stop_generating.load(Ordering::SeqCst) {
                stop_generating.store(false, Ordering::SeqCst);
                log::warn!("Request cancelled");
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };

    if use_streaming {
        let mut stream = client.generate_content_stream(&gemini_request).await?;

        log::info!("Reading Code Assist stream response...");

        loop {
            tokio::select! {
                _ = check_cancellation() => {
                    log::info!("Code Assist generation cancelled.");
                    break;
                }
                next_item = futures::StreamExt::next(&mut stream) => {
                    match next_item {
                        Some(Ok(res)) => {
                            if let Some(usage) = res.usage_metadata {
                                final_usage = Some(usage);
                            }
                            if let Some(candidate) = res.candidates.first() {
                                if let Some(parts) = &candidate.content.parts {
                                    for part in parts {
                                        handle.send((index, ChatProgress::Part(part.clone())));
                                        if let Part::Text { text, .. } = part {
                                            response_text += &text;
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => return Err(e.into()),
                        None => break,
                    }
                }
            }
        }
    } else {
        tokio::select! {
            _ = check_cancellation() => {
                log::info!("Code Assist generation cancelled.");
            }
            result = client.generate_content(&gemini_request) => {
                match result {
                    Ok(response) => {
                        final_usage = response.usage_metadata;
                        if let Some(candidate) = response.candidates.first() {
                            if let Some(parts) = &candidate.content.parts {
                                for part in parts {
                                    handle.send((index, ChatProgress::Part(part.clone())));
                                    if let Part::Text { text, .. } = part {
                                        response_text += text;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
    }

    handle.success((index, response_text, final_usage));
    Ok(())
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub enum ChatExportFormat {
    #[default]
    Plaintext,
    Json,
    Ron,
}

impl std::fmt::Display for ChatExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ChatExportFormat {
    pub const ALL: [Self; 3] = [Self::Plaintext, Self::Json, Self::Ron];

    #[inline]
    pub const fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Plaintext => &["txt"],
            Self::Json => &["json"],
            Self::Ron => &["ron"],
        }
    }
}

pub async fn export_messages(
    messages: Vec<Message>,
    format: ChatExportFormat,
    task: impl std::future::Future<Output = Option<rfd::FileHandle>>,
) -> Result<egui_notify::Toast> {
    let Some(file) = task.await else {
        log::info!("export cancelled");
        return Ok(egui_notify::Toast::info("Export cancelled"));
    };
    log::info!(
        "exporting {} messages to {file:?} (format: {format:?})...",
        messages.len()
    );

    let f = std::fs::File::create(file.path())?;
    let mut f = std::io::BufWriter::new(f);

    match format {
        ChatExportFormat::Plaintext => {
            for msg in &messages {
                writeln!(
                    f,
                    "{} - {:?} ({}): {}",
                    msg.time.to_rfc3339(),
                    msg.role,
                    msg.model,
                    msg.content
                )?;
            }
        }
        ChatExportFormat::Json => {
            serde_json::to_writer_pretty(&mut f, &messages)?;
        }
        ChatExportFormat::Ron => {
            ron::Options::default().to_io_writer_pretty(&mut f, &messages, Default::default())?;
        }
    }

    f.flush().context("failed to flush writer")?;

    log::info!("export complete");
    Ok(egui_notify::Toast::success(format!(
        "Exported {} messages to {}",
        messages.len(),
        file.file_name(),
    )))
}

fn make_summary(prompt: &str) -> String {
    const MAX_SUMMARY_LENGTH: usize = 24;
    let mut summary = String::with_capacity(MAX_SUMMARY_LENGTH);
    for (i, ch) in prompt.chars().enumerate() {
        if i >= MAX_SUMMARY_LENGTH {
            summary.push('…');
            break;
        }
        if ch == '\n' {
            break;
        }
        if i == 0 {
            summary += &ch.to_uppercase().to_string();
        } else {
            summary.push(ch);
        }
    }
    summary
}

#[derive(Debug, Clone, Copy)]
pub enum ChatAction {
    None,
    PickFiles { id: usize },
}

impl Chat {
    #[inline]
    pub fn new(id: usize, model_picker: ModelPicker) -> Self {
        Self {
            flower: CompletionFlower::new(id),
            model_picker,
            ..Default::default()
        }
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.flower.id()
    }

    fn send_message(&mut self, settings: &Settings) {
        if self.chatbox.is_empty() && self.files.is_empty() {
            return;
        }

        // remove old error messages
        self.messages.retain(|m| !m.is_error);

        let prompt = self.chatbox.trim_end().to_string();
        let model = self.model_picker.selected;
        self.messages
            .push(Message::user(prompt.clone(), model, self.files.clone()));

        if self.summary.is_empty() {
            self.summary = make_summary(&prompt);
        }

        self.chatbox.clear();
        self.files.clear();

        self.messages.push(Message::assistant(String::new(), model));

        self.spawn_completion(settings, None);
    }

    fn spawn_completion(&self, settings: &Settings, target_index: Option<usize>) {
        let handle = self.flower.handle();
        let stop_generation = self.stop_generating.clone();
        let mut messages = self.messages.clone();
        let index = target_index.unwrap_or(self.messages.len() - 1);

        if settings.include_thoughts_in_history {
            for msg in &mut messages {
                if msg.is_thought {
                    msg.is_thought = false;
                    msg.content.insert_str(0, "MY INNER REFLECTIONS: ");
                    msg.content
                        .push_str(r"--- end of inner reflections ---\r\n")
                }
            }
        }

        let use_streaming = settings.use_streaming;
        let public_file_upload = settings.public_file_upload;
        let generation_config = self.model_picker.get_generation_config();
        let auth_method = settings.auth_method;
        let api_key = settings.api_key.clone();
        let oauth_token = settings.oauth_token.clone();
        let project_id = settings.project_id.clone();
        let proxy_path = settings.proxy_path.clone();
        let model_picker = self.model_picker.clone();

        tokio::spawn(async move {
            handle.activate();

            match auth_method {
                crate::widgets::AuthMethod::ApiKey => {
                    if api_key.is_empty() {
                        handle.error((index, "API key not set.".to_string()));
                        return;
                    }

                    match model_picker.create_client(&api_key, proxy_path) {
                        Ok(gemini) => {
                            let _ = request_completion(
                                gemini,
                                messages,
                                &handle,
                                stop_generation,
                                index,
                                use_streaming,
                                public_file_upload,
                                generation_config,
                            )
                            .await
                            .map_err(|e| {
                                log::error!("failed to request completion: {e}");
                                handle.error((index, e.to_string()));
                            });
                        }
                        Err(e) => {
                            log::error!("failed to create client: {e}");
                            handle.error((index, format!("Failed to create client: {}", e)));
                        }
                    }
                }
                crate::widgets::AuthMethod::CodeAssist => {
                    if oauth_token.is_empty() || project_id.is_empty() {
                        handle.error((
                            index,
                            "OAuth token or Project ID not set. Please login in settings."
                                .to_string(),
                        ));
                        return;
                    }

                    let mut client =
                        gemini_code_assist_adapter::CodeAssistClient::new(oauth_token, project_id)
                            .with_model(model_picker.selected.to_string());

                    // Handshake
                    match client.load_code_assist().await {
                        Ok(effective_proj) => {
                            client.set_project_id(effective_proj);
                        }
                        Err(e) => log::warn!("Code Assist handshake failed: {e}"),
                    }

                    if let Err(e) = client.onboard_user().await {
                        log::warn!("Code Assist onboarding warning: {e}");
                    }

                    let _ = request_completion_code_assist(
                        client,
                        messages,
                        &handle,
                        stop_generation,
                        index,
                        use_streaming,
                        generation_config,
                    )
                    .await
                    .map_err(|e| {
                        log::error!("failed to request completion via Code Assist: {e}");
                        handle.error((index, e.to_string()));
                    });
                }
            }
        });
    }

    fn regenerate_response(&mut self, settings: &Settings, idx: usize) {
        // todo: regenerate works weird
        self.messages[idx].content = self.prepend_buf.clone();
        self.prepend_buf.clear();

        self.spawn_completion(settings, Some(idx));
    }

    fn show_chatbox(
        &mut self,
        ui: &mut egui::Ui,
        is_max_height: bool,
        is_generating: bool,
        settings: &Settings,
    ) -> ChatAction {
        let mut action = ChatAction::None;
        if let Some(idx) = self.retry_message_idx.take() {
            self.chatbox = self.messages[idx - 1].content.clone();
            self.files = self.messages[idx - 1].files.clone();
            self.messages.remove(idx);
            self.messages.remove(idx - 1);
            self.send_message(settings);
        }

        if is_max_height {
            ui.add_space(8.0);
        }

        let images_height = if !self.files.is_empty() {
            ui.add_space(8.0);
            let height = ui
                .horizontal(|ui| {
                    crate::file_handler::show_files(ui, &mut self.files, true);
                })
                .response
                .rect
                .height();
            height + 16.0
        } else {
            0.0
        };

        ui.horizontal_centered(|ui| {
            if ui
                .add(
                    egui::Button::new("➕")
                        .min_size(vec2(32.0, 32.0))
                        .corner_radius(CornerRadius::same(u8::MAX)),
                )
                .on_hover_text_at_pointer("Pick files")
                .clicked()
            {
                action = ChatAction::PickFiles { id: self.id() };
            }
            ui.with_layout(
                Layout::left_to_right(Align::Center).with_main_justify(true),
                |ui| {
                    let Self {
                        chatbox_highlighter: highlighter,
                        ..
                    } = self;
                    let mut layouter = |ui: &egui::Ui, buffer: &dyn egui::TextBuffer, wrap_width: f32| {
                        let easymark = buffer.as_str();
                        let mut layout_job = highlighter.highlight(ui.style(), easymark);
                        layout_job.wrap.max_width = wrap_width;
                        // ui.fonts(|f| f.layout_job(layout_job)) // todo
                    };

                    let text_edit_resp = ui.add(
                        egui::TextEdit::multiline(&mut self.chatbox)
                            .return_key(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter))
                            .hint_text("Ask me anything…")
                            // .layouter(&mut layouter) // todo that
                            .lock_focus(true)
                            .desired_width(f32::INFINITY),
                    );

                    self.chatbox_height = text_edit_resp.rect.height() + images_height;

                    if !is_generating
                        && text_edit_resp.has_focus()
                        && ui.input(|i| i.key_pressed(Key::Enter) && i.modifiers.is_none())
                    {
                        self.send_message(settings);
                    }
                },
            );
        });

        if is_max_height {
            ui.add_space(8.0);
        }

        action
    }

    #[inline]
    pub fn flower_active(&self) -> bool {
        self.flower.is_active()
    }

    pub fn poll_flower(&mut self, modal: &mut Modal) {
        let mut last_processed_idx = self.messages.len().saturating_sub(1);

        self.flower
            .extract(|(idx, progress)| {
                last_processed_idx = idx;

                // Clear status message when receiving new parts
                if let ChatProgress::Part(_) = progress {
                    if let Some(message) = self.messages.get_mut(idx) {
                        message.status_message = None;
                    }
                }

                match progress {
                    ChatProgress::Status { message } => {
                        if let Some(msg) = self.messages.get_mut(idx) {
                            msg.status_message = Some(message);
                        }
                    }
                    ChatProgress::FileUploading { path } => {
                        if let Some(msg) = self.messages.get_mut(idx) {
                            if let Some(attachment) =
                                msg.files.iter_mut().find(|a| a.path == path)
                            {
                                attachment.state = AttachmentState::Uploading;
                            }
                        }
                    }
                    ChatProgress::FileUploaded { path, file } => {
                        if let Some(msg) = self.messages.get_mut(idx) {
                            if let Some(attachment) =
                                msg.files.iter_mut().find(|a| a.path == path)
                            {
                                log::info!(
                                    "Updating attachment state to Uploaded for {}",
                                    path.display()
                                );
                                attachment.state = AttachmentState::Uploaded(file);
                            }
                        }
                    }
                    ChatProgress::Part(part) => {
                        match part {
                            Part::Text { text, thought, .. } => {
                                // Safely use unwrap, as we always add
                                // a placeholder message in send_message before running.
                                let current_response_msg = self.messages.last_mut().unwrap();

                                if thought.unwrap_or(false) {
                                    // This is a thought
                                    if !current_response_msg.is_thought {
                                        // If this is the first part of a "thought", turn our
                                        // placeholder message into a full "thought" message.
                                        current_response_msg.is_thought = true;
                                    }
                                    // Just append the "thought" text.
                                    current_response_msg.content.push_str(&text);
                                } else {
                                    if current_response_msg.is_thought {
                                        // "Thoughts" have just ended. Turn off the spinner for them.
                                        current_response_msg.is_generating = false;
                                        current_response_msg.generation_time =
                                            Some(current_response_msg.requested_at.elapsed());

                                        // And create a NEW, separate message for the final answer.
                                        // This will keep the thought block on screen.
                                        let model = current_response_msg.model;
                                        let mut answer_message = Message::assistant(text.into(), model);
                                        answer_message.is_generating = true; // It has its own spinner.
                                        self.messages.push(answer_message);
                                    } else {
                                        // Either there were no "thoughts", or this is a continuation of the answer.
                                        // Just append the text to the current last message.
                                        current_response_msg.content.push_str(&text);
                                    }
                                }
                            }
                            _ => {} // Handle other parts if needed
                        }
                    }
                }
            })
            .finalize(|result| {
                if let Ok((idx, _, usage)) = result {
                    if let Some(message) = self.messages.get_mut(idx) {
                        message.usage = usage;
                        message.status_message = None;
                    }
                } else if let Err(e) = result {
                    let (idx, msg) = match e {
                        Compact::Panicked(e) => {
                            (self.messages.len() - 1, format!("Tokio task panicked: {e}"))
                        }
                        Compact::Suppose((idx, e)) => (idx, e),
                    };

                    // Robust answer extraction
                    let final_msg = if let Some(start_idx) = msg.find('{') {
                        let json_candidate = &msg[start_idx..];
                        let end_idx = json_candidate.rfind('}').map(|i| i + 1).unwrap_or(json_candidate.len());
                        let json_str = &json_candidate[..end_idx];

                        let unescaped = json_str.replace("\\\"", "\"").replace("\\n", "\n");
                        let target_json = if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                            json_str
                        } else {
                            &unescaped
                        };

                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(target_json) {
                            if let Some(err_obj) = json.get("error") {
                                let code = err_obj.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
                                let status = err_obj.get("status").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
                                let message = err_obj.get("message").and_then(|v| v.as_str()).unwrap_or("No message");

                                match status {
                                    "RESOURCE_EXHAUSTED" => {
                                        let retry_info = if message.contains("retry in ") {
                                            if let Some(pos) = message.find("retry in ") {
                                                format!("\n\n⏳ **Suggestion:** {}", &message[pos..])
                                            } else {
                                                String::new()
                                            }
                                        } else {
                                            String::new()
                                        };

                                        format!("🛑 **Quota Exhausted (429)**\n\nYou've hit the Gemini API rate limit. Please wait a bit or check your Google AI Studio quota.{}", retry_info)
                                    },                                                                    "NOT_FOUND" => {
                                        format!("🚫 **Model Not Found (404)**\n\nThe model you selected is either not found or not supported for this operation. Try choosing a different model.\n\n**Details:** {}", message)
                                    },
                                    "PERMISSION_DENIED" => {
                                        format!("🔒 **Permission Denied (403)**\n\nCheck your API Key and project permissions. Make sure the Key is valid for the selected region.\n\n**Details:** {}", message)
                                    },
                                    "INVALID_ARGUMENT" => {
                                        format!("❌ **Invalid Request (400)**\n\nSomething is wrong with the request parameters.\n\n**Details:** {}", message)
                                    },
                                    _ => format!("❗ **Gemini API Error ({})**\n\n**Status:** {}\n**Message:** {}", code, status, message)
                                }                            } else {
                                    serde_json::to_string_pretty(&json).unwrap_or_else(|_| msg.clone())
                                }
                        } else {
                            msg.clone()
                        }
                    } else {
                        msg.clone()
                    };

                    if let Some(message) = self.messages.get_mut(idx) {
                        message.content = final_msg.clone();
                        message.is_error = true;
                        message.is_generating = false;
                        message.generation_time = Some(message.requested_at.elapsed());
                        message.status_message = None;
                    }

                    modal
                        .dialog()
                        .with_body(final_msg)
                        .with_title("Failed to generate completion!")
                        .with_icon(Icon::Error)
                        .open();
                }

                if let Some(last_msg) = self.messages.last_mut() {
                    if last_msg.is_generating {
                        last_msg.is_generating = false;
                        last_msg.generation_time = Some(last_msg.requested_at.elapsed());
                        last_msg.status_message = None;
                    }
                }
            });
    }

    pub fn last_message_contents(&self) -> Option<String> {
        for message in self.messages.iter().rev() {
            if message.content.is_empty() {
                continue;
            }
            return Some(if message.is_user() {
                format!("You: {}", message.content)
            } else {
                message.content.to_string()
            });
        }
        None
    }

    fn stop_generating_button(&self, ui: &mut egui::Ui, radius: f32, pos: Pos2) {
        let rect = Rect::from_min_max(pos + vec2(-radius, -radius), pos + vec2(radius, radius));
        let (hovered, primary_clicked) = ui.input(|i| {
            (
                i.pointer
                    .interact_pos()
                    .map(|p| rect.contains(p))
                    .unwrap_or(false),
                i.pointer.primary_clicked(),
            )
        });
        if hovered && primary_clicked {
            self.stop_generating.store(true, Ordering::SeqCst);
        } else {
            ui.painter().circle(
                pos,
                radius,
                if hovered {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    if ui.style().visuals.dark_mode {
                        let c = ui.style().visuals.faint_bg_color;
                        Color32::from_rgb(c.r(), c.g(), c.b())
                    } else {
                        Color32::WHITE
                    }
                } else {
                    ui.style().visuals.window_fill
                },
                Stroke::new(2.0, ui.style().visuals.window_stroke.color),
            );
            ui.painter().rect_stroke(
                rect.shrink(radius / 2.0 + 1.2),
                2.0,
                Stroke::new(2.0, Color32::DARK_GRAY),
                egui::StrokeKind::Outside,
            );
        }
    }

    fn show_chat_scrollarea(
        &mut self,
        ui: &mut egui::Ui,
        settings: &Settings,
        commonmark_cache: &mut CommonMarkCache,
        #[cfg(feature = "tts")] tts: SharedTts,
    ) -> Option<usize> {
        let mut new_speaker: Option<usize> = None;
        let mut any_prepending = false;
        let mut regenerate_response_idx = None;
        let mut message_to_delete_idx: Option<usize> = None;
        egui::ScrollArea::vertical()
            .animated(false)
            .stick_to_bottom(true)
            .auto_shrink(false)
            .show(ui, |ui| {
                let scrollbar_width = ui.style().spacing.scroll.bar_width + 12.0;
                ui.set_width(ui.available_width() - scrollbar_width);
                

                // ui.add_space(16.0);
                RobustVirtualScroll::new(Id::new("chat_virtual_list"))
                    // todo: anchors
                    // .anchors(anch_indices, |index| { // TODO! maybe any ref?
                    //     anchors_map.get(&index).cloned().unwrap_or_default() // bruh
                    // })
                    .show(
                        ui,
                        self.messages.len(),
                        |i| Id::new(i), // todo: add ID in struct?!
                        |ui, index|
                {
                    // println!("Rendering: '{index}'");
                    let message = &mut self.messages[index]; // надо
                    let prev_speaking = message.is_speaking;

                    if any_prepending && message.is_prepending {
                        message.is_prepending = false;
                    }

                    ui.push_id(index, |ui| {
                        let action = message.show(
                            ui,
                            commonmark_cache,
                            #[cfg(feature = "tts")]
                            tts.clone(),
                            index,
                            &mut self.prepend_buf,
                        );
                        match action {
                            MessageAction::None => (),
                            MessageAction::Retry(idx) => {
                                self.retry_message_idx = Some(idx);
                            }
                            MessageAction::Regenerate(idx) => {
                                regenerate_response_idx = Some(idx);
                            }
                            MessageAction::Delete(idx) => {
                                message_to_delete_idx = Some(idx);
                            }
                        }
                    });

                    any_prepending |= message.is_prepending;

                    if !prev_speaking && message.is_speaking {
                        new_speaker = Some(index);
                    }
                });

                ui.add_space(12.0);
            });
        if let Some(regenerate_idx) = regenerate_response_idx {
            self.regenerate_response(settings, regenerate_idx);
        }
        if let Some(idx) = message_to_delete_idx {
            self.messages.remove(idx);
        }
        new_speaker
    }

    fn send_text(&mut self, settings: &Settings, text: &str) {
        self.chatbox = text.to_owned();
        self.send_message(settings);
    }

    fn show_suggestions(&mut self, ui: &mut egui::Ui, settings: &Settings) {
        // todo broken weird shit :p
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            widgets::centerer(ui, |ui| {
                let avail_width = ui.available_rect_before_wrap().width() - 24.0;
                ui.horizontal(|ui| {
                    ui.heading(format!(
                        "{}",
                        self.model_picker.selected.to_string().replace("-", " ")
                    )); // todo improve it
                });
                egui::Grid::new("suggestions_grid")
                    .num_columns(3)
                    .max_col_width((avail_width / 2.0).min(200.0))
                    .spacing(vec2(6.0, 6.0))
                    .show(ui, |ui| {
                        // TODO change it
                        if widgets::suggestion(ui, "Tell me a fun fact", "about the Roman empire")
                            .clicked()
                        {
                            self.send_text(settings, "Tell me a fun fact about the Roman empire");
                        }
                        if widgets::suggestion(
                            ui,
                            "Show me a code snippet",
                            "of a web server in Rust",
                        )
                        .clicked()
                        {
                            self.send_text(
                                settings,
                                "Show me a code snippet of a web server in Rust",
                            );
                        }
                        widgets::dummy(ui);
                        ui.end_row();

                        if widgets::suggestion(ui, "Tell me a joke", "about crabs").clicked() {
                            self.send_text(settings, "Tell me a joke about crabs");
                        }
                        if widgets::suggestion(ui, "Give me ideas", "for a birthday present")
                            .clicked()
                        {
                            self.send_text(settings, "Give me ideas for a birthday present");
                        }
                        widgets::dummy(ui);
                        ui.end_row();
                    });
            });
        });
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        settings: &Settings,
        #[cfg(feature = "tts")] tts: SharedTts,
        #[cfg(feature = "tts")] stopped_speaking: bool,
        commonmark_cache: &mut CommonMarkCache,
    ) -> ChatAction {
        let avail = ctx.available_rect();
        let max_height = avail.height() * 0.4 + 24.0;
        let chatbox_panel_height = self.chatbox_height + 24.0;
        let actual_chatbox_panel_height = chatbox_panel_height.min(max_height);
        let is_generating = self.flower_active();
        let mut action = ChatAction::None;

        egui::TopBottomPanel::bottom("chatbox_panel")
            .exact_height(actual_chatbox_panel_height)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    action = self.show_chatbox(
                        ui,
                        chatbox_panel_height >= max_height,
                        is_generating,
                        settings,
                    );
                });
            });

        #[cfg(feature = "tts")]
        let mut new_speaker: Option<usize> = None;

        egui::CentralPanel::default()
            .frame(Frame::central_panel(&ctx.style()).inner_margin(Margin {
                left: 16,
                right: 16,
                top: 0,
                bottom: 3,
            }))
            .show(ctx, |ui| {
                // ui.ctx().set_debug_on_hover(true); // TODO DEBUG
                if self.messages.is_empty() {
                    self.show_suggestions(ui, settings);
                } else {
                    #[allow(unused_variables)]
                    if let Some(new) = self.show_chat_scrollarea(
                        ui,
                        settings,
                        commonmark_cache,
                        #[cfg(feature = "tts")]
                        tts,
                    ) {
                        #[cfg(feature = "tts")]
                        {
                            new_speaker = Some(new);
                        }
                    }

                    // stop generating button
                    if is_generating {
                        self.stop_generating_button(
                            ui,
                            16.0,
                            pos2(
                                ui.cursor().max.x - 32.0,
                                avail.height() - 32.0 - actual_chatbox_panel_height,
                            ),
                        );
                    }
                }
            });

        #[cfg(feature = "tts")]
        {
            if let Some(new_idx) = new_speaker {
                log::debug!("new speaker {new_idx} appeared, updating message icons");
                for (i, msg) in self.messages.iter_mut().enumerate() {
                    if i == new_idx {
                        continue;
                    }
                    msg.is_speaking = false;
                }
            }
            if stopped_speaking {
                log::debug!("TTS stopped speaking, updating message icons");
                for msg in self.messages.iter_mut() {
                    msg.is_speaking = false;
                }
            }
        }

        action
    }
}
