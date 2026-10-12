// SPDX-License-Identifier: Apache-2.0
use facts_yolo::{ffi::types::*, fy_abi_query};
use std::ptr::null_mut;

#[test]
fn build_id_matches_ci_checkout_or_local_override() {
    let expected = if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true") {
        std::env::var("GITHUB_SHA").expect("CI must identify its exact checkout")
    } else {
        option_env!("FACTSYOLO_BUILD_ID")
            .unwrap_or("local-unversioned")
            .to_owned()
    };
    let mut out = FyAbiInfo::default();
    assert_eq!(unsafe { fy_abi_query(&mut out, null_mut()) }, FY_OK);
    let end = out.build_id.iter().position(|b| *b == 0).unwrap();
    assert_eq!(&out.build_id[..end], expected.as_bytes());
}
