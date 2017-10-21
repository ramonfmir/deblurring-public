#!/usr/bin/env ruby

# Sends an image to a channel.
# Usage: slack_file_upload <filename> "<channel>" 

require 'json'
require 'shellwords'

filename = ARGV[0].to_s
channel = ARGV[1].to_s
token = "xoxp-253117849458-253045350483-260796961334-84b69e105dba56f48b37acdf52001e54"

output = `curl -F file=@#{Shellwords.escape filename} \
               -F channels=#{channel} \
               -F token=#{token} \
               https://deblurring.slack.com/api/files.upload`

result = JSON.parse(output)

if not result['ok'] then
 puts "Upload failed: #{result['error']}"
 exit 1
end
