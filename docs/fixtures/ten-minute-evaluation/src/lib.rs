pub fn greeting(name: &str) -> String {
    format!("Hello, {name}!")
}

pub fn welcome(name: &str) -> String {
    greeting(name)
}
