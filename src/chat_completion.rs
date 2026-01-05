use crate::chat::{ChatProgress, CompletionFlowerHandle, Message, MessageRole};
use crate::file_handler::{convert_file_to_part, Attachment, AttachmentState, FileResult};
use anyhow::Result;
use gemini_rust::{Content, FileData, Gemini, Part, Role};

pub async fn build_history(
    gemini: &Gemini,
    messages: &[Message],
    extra_content: Option<(&str, &[Attachment])>,
    public_file_upload: bool,
    status_channel: Option<(usize, &CompletionFlowerHandle)>,
) -> Result<Vec<Content>> {
    let mut history: Vec<Content> = Vec::new();
    let mut parts_buffer: Vec<Part> = Vec::new();
    let mut active_role: Option<Role> = None;

    // Process main messages
    for (msg_idx, message) in messages.iter().enumerate() {
        if message.is_thought || (message.content.is_empty() && message.files.is_empty()) {
            continue;
        }

        let message_role = match message.role {
            MessageRole::User => Role::User,
            MessageRole::Assistant => Role::Model,
        };

        if let Some(current_role) = &active_role {
            if *current_role != message_role {
                if !parts_buffer.is_empty() {
                    history.push(Content {
                        parts: Some(std::mem::take(&mut parts_buffer)),
                        role: Some(current_role.clone()),
                    });
                }
                active_role = Some(message_role);
            }
        } else {
            active_role = Some(message_role);
        }

        process_attachments(
            gemini,
            &message.files,
            &mut parts_buffer,
            public_file_upload,
            status_channel,
            msg_idx,
        )
        .await;

        if !message.content.is_empty() {
            parts_buffer.push(Part::Text {
                text: message.content.clone(),
                thought: None,
                thought_signature: None,
            });
        }
    }

    if !parts_buffer.is_empty() {
        if let Some(role) = active_role {
            history.push(Content {
                parts: Some(std::mem::take(&mut parts_buffer)),
                role: Some(role),
            });
        }
    }

    // Process extra content (e.g. from input box)
    if let Some((text, files)) = extra_content {
        let mut extra_parts = Vec::new();
        process_attachments(
            gemini,
            files,
            &mut extra_parts,
            false, // Don't upload extra content files (usually local for preview/counting)
            None,  // No status updates for extra content (usually used for counting)
            0,     // Index irrelevant when status_channel is None
        )
        .await;

        if !text.is_empty() {
            extra_parts.push(Part::Text {
                text: text.to_string(),
                thought: None,
                thought_signature: None,
            });
        }

        if !extra_parts.is_empty() {
            history.push(Content {
                parts: Some(extra_parts),
                role: Some(Role::User),
            });
        }
    }

    // Clear status message if we have a handle
    if let Some((index, h)) = status_channel {
        h.send((
            index,
            ChatProgress::Status {
                message: String::new(),
            },
        ));
    }

    Ok(history)
}

async fn process_attachments(
    gemini: &Gemini,
    files: &[Attachment],
    parts_buffer: &mut Vec<Part>,
    allow_upload: bool,
    status_channel: Option<(usize, &CompletionFlowerHandle)>,
    file_msg_index: usize,
) {
    for attachment in files {
        let file_path = &attachment.path;
        let filename = file_path
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();

        if let AttachmentState::Uploaded(remote_file) = &attachment.state {
            let is_expired = if let Some(exp) = remote_file.expiration_time {
                exp < time::OffsetDateTime::now_utc()
            } else {
                false
            };

            if !is_expired {
                if let Some(uri) = &remote_file.uri {
                    parts_buffer.push(Part::FileData {
                        file_data: FileData {
                            file_uri: uri.to_string(),
                            mime_type: remote_file.mime_type.clone().unwrap_or_default(),
                        },
                    });
                    continue;
                }
            }
        }

        if let Some((status_idx, h)) = status_channel {
            h.send((
                status_idx,
                ChatProgress::Status {
                    message: format!("Processing file: {filename}..."),
                },
            ));
            // Trigger Uploading state in UI (target the message with the file)
            if allow_upload {
                h.send((
                    file_msg_index,
                    ChatProgress::FileUploading {
                        path: file_path.clone(),
                    },
                ));
            }
        }

        // If status_channel is None (e.g. counting), force inline (upload=false)
        let effective_upload = allow_upload && status_channel.is_some();

        match convert_file_to_part(gemini, file_path, effective_upload).await {
            Ok(FileResult::InlinePart(part)) => parts_buffer.push(part),
            Ok(FileResult::UploadedFile(file_handle)) => {
                if let Some((_, h)) = status_channel {
                    h.send((
                        file_msg_index,
                        ChatProgress::FileUploaded {
                            path: file_path.clone(),
                            file: file_handle.get_file_meta().clone(),
                        },
                    ));
                }

                if let Ok(file_data) = FileData::try_from(&file_handle) {
                    parts_buffer.push(Part::FileData { file_data });
                }
            }
            Err(e) => log::error!("Failed to process file {}: {}", file_path.display(), e),
        }
    }
}
