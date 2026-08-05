use std::time::Duration;

use brokk_anvil_sdk::apis::{
    configuration::Configuration,
    models_api::{self, ListModelsParams},
    runs_api::{self, CreateRunParams, GetRunParams, ListRunsParams},
    server_api, sessions_api, tools_api,
};
use brokk_anvil_sdk::models::{
    CreateRunRequest, CreateSessionRequest, create_session_request::PermissionMode, run::Status,
};

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let base = args.next().expect("base URL argument");
    let cwd = args.next().expect("workspace argument");
    assert!(args.next().is_none(), "unexpected extra arguments");

    let mut configuration = Configuration::default();
    configuration.base_path = base;
    configuration.user_agent = Some("brokk-anvil-sdk-conformance".to_owned());

    assert_eq!(
        server_api::get_health(&configuration)
            .await
            .expect("Rust SDK health")
            .status,
        brokk_anvil_sdk::models::health::Status::Ok
    );
    models_api::list_models(&configuration, ListModelsParams { refresh: None })
        .await
        .expect("Rust SDK models");
    tools_api::list_tools(&configuration)
        .await
        .expect("Rust SDK tools");

    let mut request = CreateSessionRequest::new(cwd);
    request.permission_mode = Some(PermissionMode::AcceptEdits);
    let session = sessions_api::create_session(
        &configuration,
        sessions_api::CreateSessionParams {
            create_session_request: request,
        },
    )
    .await
    .expect("Rust SDK create session");
    let run = runs_api::create_run(
        &configuration,
        CreateRunParams {
            session_id: session.id.clone(),
            create_run_request: CreateRunRequest::new("Rust SDK conformance turn".to_owned()),
        },
    )
    .await
    .expect("Rust SDK create run");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let terminal = loop {
        let current = runs_api::get_run(
            &configuration,
            GetRunParams {
                run_id: run.id.clone(),
            },
        )
        .await
        .expect("Rust SDK get run");
        if current.status != Status::Running {
            break current;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "Rust SDK run timed out"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert_eq!(terminal.status, Status::Completed);
    assert_eq!(
        terminal.result_text.as_deref(),
        Some("SDK conformance complete")
    );

    let runs = runs_api::list_runs(
        &configuration,
        ListRunsParams {
            session_id: session.id.clone(),
        },
    )
    .await
    .expect("Rust SDK list runs");
    assert!(runs.runs.iter().any(|candidate| candidate.id == run.id));
    assert!(
        sessions_api::delete_session(
            &configuration,
            sessions_api::DeleteSessionParams {
                session_id: session.id,
            },
        )
        .await
        .expect("Rust SDK delete session")
        .deleted
    );
    println!("Rust SDK conformance passed");
}
