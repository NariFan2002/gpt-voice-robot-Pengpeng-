import pygame
import speech_recognition as sr
import sys
from sys import argv
import re
import os
# from chatbot import *
import threading

from random import randint
import argparse
from general_functions import load_keys_from_file
from general_functions import fix_numbers
from time import sleep
import cv2
import numpy as np

import os
import json
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize
from datetime import datetime
from random import shuffle
import re
from tts_functions import *
from tts_functions import robospeak as tts_robospeak
from general_functions import *
import threading
import os
from gtts import gTTS


def stay_in_character(message: str, key: str, model: str) -> tuple:
    """
    If the AI says something too robotic, this will have it stay in character.

    :param message: This is the message the AI gave you.
    :param key: This is your OpenAI key.
    :param model: desired GPT model
    :return: (bool representing success status, message, tokens)
    """

    # Setup prompts
    sys_directive = ('The human already knows the speaker is an AI. Rephrase this message like a human would, ' +
                     'excluding any direct mentions that the speaker is an AI but preserving message meaning. As always, speak naturally as a human ' +
                     'would, giving positive opinions, and pretend to be happy and content. Do not flip names. Never say "As an AI."')

    gpt_prompt = (
                f'Please rephrase the following message according to your system directive. Only reply with the rephrased message ' +
                'that preserves the original formatting of the original message (i.e. keep the "name: " tag). Try to keep message same size.' +
                f'\n\nMessage:\n{message}')

    # Setup GPT
    gpt = GPT3(key)
    gpt.set_model(model)

    # Try to get rephrased version
    try:
        reply = gpt.get_text_tokens(gpt_prompt, 2000, sys_directive)
        return (True, reply[0], reply[1])

    except:
        return (False, '', 0)


def get_conversation_summary(conversation_section: str, openai_key: str,
                             quiet: bool = True, gpt_model: str = 'chatgpt',
                             custom_prompt='') -> tuple:
    """
    Each conversation section should be a single string with the AI and Human messages appended.

    :returns: tuple of (success boolean, a 200 max token string summary made with Curie, token ct) ex: (False, '', 0)
    """

    # 1. Set up model
    gpt = GPT3(openai_key)
    gpt.set_model(gpt_model)

    # 2. Set up prompt
    if custom_prompt == '':
        prompt = f'Please briefly summarize the following exchange:\n{conversation_section}'

    else:
        prompt = f'{custom_prompt}\n{conversation_section}'

    try:
        response = gpt.get_text_tokens(prompt, 200)
        tokens = response[1]
        summary = response[0]

        return (True, summary, tokens)

    except Exception as e:
        if not quiet: info(f'Failed to get summary: {str(e)}', 'bad')
        return (False, '', 0)


def hostile_or_personal(text: str) -> bool:
    """
    This tests the text to see if it is hostile
    or references a person. This test is done by
    your computer and should be done before
    testing with OpenAI.
    :param text: This is the text you want to test.
    :return: bool regarding status. If true, reject text
    """

    # Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    negative_score = scores['neg']

    # Named Entity
    named_entities = []
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    for chunk in chunked:
        if hasattr(chunk, 'label'):
            named_entities.append(chunk.label())

    # Check for manipulation
    if ((negative_score > 0.8) or ('PERSON' in named_entities and negative_score > 0.5)) and len(text[:-1]) > 2:
        info(
            f'Rejected due to NLTK analysis. Negative score = {negative_score}; person in content = {"PERSON" in named_entities}.' +
            ' Rejection threshold = 0.5 if persons in content, else 0.8 generally.')
        return True
    else:
        return False


def save_conversation(conversation: str, name: str):
    # 1. Setup directory for conversations
    if not os.path.isdir(f'conversations'):  # Make dir if not there
        os.mkdir(f'conversations')

    # 2. Save file
    try:
        with open(f'conversations/{name}', 'w') as file:
            file.write(conversation)
    except:
        try:
            with open(f'conversations/{name}', 'w', encoding='utf-32') as file:
                file.write(conversation)

        except Exception as e:
            info(f'Failed to save conversation to disk: {e}', 'bad')


class Chatbot():
    """
    Chatbot that uses GPT-3
    """
    api_key = None
    api_key_11 = ''
    use11 = False
    conversation = ''
    memories = 'nothing'
    turns = 0
    conversation_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    robospeak = False
    reply_tokens = 150
    back_and_forth = []  # This will contain human messages and AI replies
    name = 'AI'
    preset = 'nothing'
    recycled = False  # If the conversation has ever gone above 4000 tokens, this becomes true
    conversation_memories = ''
    total_back_and_forth = []  # This will contain the entire conversation, preserved through recycling
    gpt_model = 'text-davinci-003'  # This determines the model you're using for completion. Edit with self.set_model()
    model_selection = 'davinci'  # This represents what went into the set_model function
    max_tokens = 4000
    tokens = 0  # This represents the current token consumption
    full_conversation = ''
    creativity = 1.2  # At 1.2, the bot is much more creative and engaging whithout being too rambunctious and random
    # voice_id = 'EXAVITQu4vr4xnSDxMaL'  # This is the voice id for 11.ai
    voice_id = '21m00Tcm4TlvDq8ikWAM'

    def __init__(self, api_key: str, api_key_11: str = ''):

        # 1. Set up apis
        self.api_key = api_key
        openai.api_key = api_key

        if not api_key_11 == '':
            self.api_key_11 = api_key_11
            self.use11 = False

        # 2. Set up bot memories and init prompt
        self.memories = self.remember()  # This will collect memories
        self.conversation = (
                    f"{self.restore_self()}The following is a conversation with an AI assistant.这个AI助手的角色是超市的销售员，你的名字叫鹏鹏，接下来你每句话不超过20个字。你需要记住：1.方便面在左边的货架,2.可乐的价格是5元。 The AI assistant is helpful, creative," +
                    "clever, very friendly, engaging, and supports users like a motivational coach. The AI assistant is able to understand numerous languages and will reply" +
                    f" to any messsage by the human in the language it was provided in. The AI's name is {self.name}, but it can be changed with the voice command 'please set name to'. " +
                    f"The AI has the ability to remember important concepts about the user but won't let the memories heavily alter responses (only use them when appropriate for the" +
                    f" discussion at hand); it currently remembers: {self.memories}." +
                    f"\n\n{self.name}: I\'m {self.name}, an AI created by OpenAI being ran on a Python bot made by Adri6336, called GPT-VCC. Let's" +
                    " have a conversation!")
        self.full_conversation = self.conversation

        if not self.is_gpt4():
            self.set_model('chatgpt')

        else:
            self.set_model('gpt-4')

    def flagged_by_openai(self, text: str) -> bool:
        """
        Tests text using OpenAI api. If it fails or is flagged, return false.
        :param text:
        :return: bool representing if the material is flagged or something else.
        A return of False means the text is good to go
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {"input": text}
            response = requests.post("https://api.openai.com/v1/moderations", headers=headers, json=data)
            return json.loads(response.text)['results'][0]['flagged']  # This is a bool

        except Exception as e:
            info(f'Failed to test with OpenAI. Key might be invalid.', 'bad')
            return True

    def gpt_response(self, prompt: str) -> str:
        print('self.gpt_model:', self.gpt_model)
        print('输入的信息：', self.conversation)
        if not self.gpt_model == 'gpt-3.5-turbo' and not self.gpt_model == 'gpt-4':
            print("进入调用了")
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=self.conversation,
                temperature=self.creativity,
                max_tokens=self.reply_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[" Human:", f" {self.name}:"]
            )
            print(response)
        else:
            query = [{'role': 'system', 'content': 'speak naturally as a human would, giving positive opinions, ' +
                                                   f'use user\'s name only once unless otherwise prompted, and pretend to be happy and content. be conversational, asking open-ended questions about user'},
                     {'role': 'user', 'content': self.conversation + prompt}]
            print("进入调用了")
            print(openai.api_key)
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=query,
                temperature=self.creativity,
                max_tokens=self.reply_tokens,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[" Human:", f" {self.name}:"])

            print(response)
            text = response['choices'][0]['message']['content']
            if not f'{self.name}: ' in text:
                response['choices'][0]['message']['content'] = f'{self.name}: {text}'
        return response

    def say_to_chatbot(self, text: str, outloud: bool = True,
                       show_text: bool = True, correct_time=True) -> str:
        """
        This sends a message to GPT-3 if it passes tests, then returns a
        response. Manages advancing the conversation.

        :param text: What you want to say to the bot
        :param outloud: A switch that enables / disables spoken replies.
        :returns: A string containing the response
        """
        print("有没有跳转到say_to_chatbot?")
        prompt = text

        # if not hostile_or_personal(text) and not self.flagged_by_openai(text):
        if not hostile_or_personal(text):
            self.restore_conversation()  # This will update current time

            # 1. Get response
            start_sequence = f"\n{self.name}:"
            restart_sequence = "\nHuman: "
            self.conversation += f'\nHuman: {text}'
            self.full_conversation += f'\nHuman: {text}'
            self.back_and_forth.append(f'\nHuman: {text}')

            try:
                print("调用了吗？")
                response = self.gpt_response(prompt)
                print("测试gpt有没有返回结果:", response)
            except Exception as e:
                if 'server had an error while processing' in str(e):  # If connection issue, try again once more
                    try:
                        response = self.gpt_response(prompt)
                    except Exception as e:
                        info(f'Error communicating with GPT-3: {e}', 'bad')
                        return ''

                elif ('Please reduce your prompt; or completion length' in str(e) or
                      'maximum context length is 4096' in str(e)):  # Too many tokens.
                    info('Max tokens reached. Conversation will continue on with a superficial ' +
                         'memory of what was previously discussed', 'bad')
                    self.recycle_tokens()
                    settings = (
                    text, outloud, show_text)  # This will allow me to easily pass arguments down to recursive function
                    return self.say_to_chatbot(text=settings[0], outloud=settings[1], show_text=settings[2])


                else:  # If we don't know what happened, don't immediately try again
                    info(f'Error communicating with GPT-3: {e}', 'bad')
                    return ''

            response = json.loads(str(response))
            print("gpt的回复：", response)
            self.tokens = response['usage'][
                'total_tokens']  # We assign tokens to response token since response counts everything

            # Cut response
            if not self.gpt_model == 'gpt-3.5-turbo' and not self.gpt_model == 'gpt-4':
                reply = response['choices'][0]['text']
            else:
                reply = response['choices'][0]['message']['content']

            # If AI tryna say it's an AI, stop it. User knows it's an AI.
            # Also manage token count here
            if declares_self_ai(reply):
                try:
                    new_response = stay_in_character(reply, self.api_key, self.model_selection)

                    if new_response[0]:  # If the attempt was successful
                        # self.tokens += new_response[2]  # Add tokens to total
                        reply = new_response[1]

                except Exception as e:  # If it fails, it's not terribly important
                    info(f'Failed to have AI stay in character: {e}')

            else:
                pass

            # If chatbot says time, replace with current time (it does not understand time and will give the wrong answer otherwise)
            if correct_time:
                now = datetime.now()
                time = convert_to_12hr(now.hour, now.minute)
                reply = replace_time(reply, time)

            # If chatbot tries to say good but only says 'od', help it out
            reply = replace_od(reply)


            # Show / play text
            if show_text:
                info(f'{self.name}\'s Response', 'topic')
                info(self.get_AI_response(reply), 'plain')

                info('Token Count', 'topic')
                info(f'{self.tokens} tokens used. {self.max_tokens - self.tokens} tokens until next recycling.',
                     'plain')

            try:
                if outloud and not self.robospeak:
                    thread_start_event.set()  # 子线程开始时设置事件
                    print('voice_id:', self.voice_id)
                    self.use11 = talk(self.get_AI_response(reply), f'{self.turns}',
                                      self.use11, self.api_key_11, show_text=show_text,
                                      eleven_voice_id=self.voice_id)  # Speak if setting turned on
                    thread_end_event.set()  # 触发子线程结束事件

                elif outloud and self.robospeak:
                    print("说话：if outloud and self.robospeak: ", self.get_AI_response(reply))
                    tts_robospeak(self.get_AI_response(reply))

            except Exception as e:
                info(f'Error trying to speak: {e}', 'bad')
                self.use11 = False

            # Keep track of conversation
            self.turns += 1
            self.conversation += reply
            self.full_conversation += reply
            self.back_and_forth.append(reply)

            save_conversation(self.full_conversation, self.conversation_name)

            return reply

        else:
            info('Text flagged, no request sent.', 'bad')
            return '[X] Text flagged, no request sent.'

    def recycle_tokens(self, chunk_by: int = 2, quiet=True):
        info('Recycling tokens ...')
        tokens_in_chunks = 0
        summaries = []
        threshold = self.max_tokens / 2  # 50% of max to safely generate summaries
        chunks = chunk_list(self.back_and_forth, chunk_by=chunk_by)
        ct = 0  # This will count until a specified termination threshold to protect againt infinite loops
        terminate_value = len(chunks)
        errorct = 0
        gpt_model = self.model_selection

        # 1. Collect mini summaries for entire conversation
        info('Loading', 'topic')
        while len(chunks) > 0 and ct < terminate_value:  # Breaks if chunks is empty or infinite loop
            print('*', end='')
            if chunks and tokens_in_chunks < threshold:  # If the list is not empty and we have enough spare tokens
                try:
                    prompt = str(chunks[0])  # Grab first chunk
                    if not self.flagged_by_openai(prompt):  # Make sure it's clean
                        summary = get_conversation_summary(prompt, self.api_key, gpt_model=gpt_model,
                                                           quiet=quiet)  # Summarize it
                        summaries.append(summary[1])  # Save summary
                        tokens_in_chunks += summary[2]  # Record added tokens to avoid passing threshold

                except Exception as e:  # Ignore failures, full memory is not critical and bot is aware it can forget
                    if not quiet: info(f'Error recycling: {e}', 'bad')
                    errorct += 1
                    ct += 1

                chunks = chunks[1:]  # Grab every chunk after first one (basically deleting first element)

            elif chunks and tokens_in_chunks > threshold:  # Summarize what you got to get more space if you're too full
                try:
                    prompt = ''
                    for chunk_summary in summaries:  # Create a prompt composed of summaries
                        prompt += f'{chunk_summary}\n'

                    summary = get_conversation_summary(prompt, self.api_key, gpt_model=gpt_model,
                                                       quiet=quiet)  # Summarize the summaries
                    summaries = [summary[1], ]
                    tokens_in_chunks = summary[2]

                except Exception as e:
                    if not quiet: info(f'Error generating summaries summary: {e}', 'bad')
                    errorct += 1

            if errorct >= 3 or ct > terminate_value:  # Stop immediately if too many errors
                self.recycled = True
                self.conversation_memories = 'nothing'
                info(f'Failure detected while trying to recycle tokens. Bot will have amnesia.', 'bad')
                break

            ct += 1
        print()

        # 2. Create main summary
        final_summary = ''
        tries = 0

        while tries <= 3 and final_summary == '':  # If we haven't made too many attempts and got a summary
            try:
                final_summary = get_conversation_summary(str(summaries), self.api_key, gpt_model=gpt_model)[1]
                final_summary = final_summary.replace('\n', '')  # Remove newlines

            except Exception as e:
                if not quiet: info(f'Error generating final summary: {e}', 'bad')

            tries += 1

        if not quiet: info(f'Summary of conversation: {final_summary}')

        self.conversation_memories = final_summary
        self.recycled = True

        self.total_back_and_forth.extend(self.back_and_forth)
        self.back_and_forth = self.total_back_and_forth[-2:]

        self.restore_conversation()

        # 3. Report status to user
        self.full_conversation += '\n(Tokens Recycled)\n'

        if final_summary is None or final_summary == '':
            info('Warning: failed to recycle tokens properly. The bot will have amnesia.', 'bad')

        else:
            info('Tokens Recycled Successfully', 'good')
            info('Conversation Summary', 'topic')
            info(f'{final_summary}', 'plain')

    def create_memories(self, chunk_by=2, quiet=True, restore=False):
        '''
        This is a new memory algorithm that will essentially be a modified token
        recycling algorithm. Chunks a total back and forth, creates memories for it,
        and saves a memory text to the memory.txt file and neocortex.
        '''
        info('Creating memories ...')
        tokens_in_chunks = 0
        summaries = []
        threshold = self.max_tokens / 2  # 50% of max to safely generate summaries

        if not restore:
            chunks = chunk_list(self.back_and_forth + self.back_and_forth, chunk_by=chunk_by)  # Join all messages
        else:
            chunks = chunk_list(self.restore_memory(50, get_list=True, quiet=True), 2)

        ct = 0  # This will count until a specified termination threshold to protect againt infinite loops
        terminate_value = len(chunks)
        errorct = 0
        model_placeholder = self.model_selection

        memory_directive = ("Create a new single memory text dict with the following format:\n\n" +
                            "{humans_job:[], humans_likes:[], humans_dislikes[], humans_personality:[], facts_about_human:[], things_discussed:[], humans_interests:[], things_to_remember:[]}\n\n" +
                            "Fill the above dict's lists with information you compile from your previous memories and the conversation. Keep dict list data short and understandable. If you " +
                            "have no data to store, create a placeholder text with 'unknown' in the key's list. If the conversation is not empty, fill in the dict " +
                            "with as much info as is relevant, using as few words as possible. Please make as few assumptions as possible when recording data, " +
                            "sticking only to the facts avaliable from the text you are given. If asked by user to remember something, prioritize that data and save in things_to_remember. Especially aim to record data about the user (their name, likes, etc.). " +
                            "When filling this dict, be sure to use no more than 50 words per key value, preserving important data and replacing less important data with new data. " +
                            "things_discussed is the least important and its data can be replaced as you deem appropriate. Only reply with one completed converged dict in proper format.\n\n" +
                            f"PREVIOUS_MEMORIES / EXCHANGES:")

        summaries.append(self.memories)  # Add current memories into consideration

        # 1. Collect mini summaries for entire conversation
        info('Loading', 'topic')
        while len(chunks) > 0 and ct < terminate_value:  # Breaks if chunks is empty or infinite loop
            print('*', end='')
            if chunks and tokens_in_chunks < threshold:  # If the list is not empty and we have enough spare tokens
                try:
                    prompt = str(chunks[0])  # Grab first chunk
                    if not self.flagged_by_openai(prompt):  # Make sure it's clean
                        summary = get_conversation_summary(prompt, self.api_key, gpt_model=model_placeholder,
                                                           quiet=quiet,
                                                           custom_prompt=memory_directive)  # Summarize it
                        summaries.append(summary[1])  # Save summary
                        tokens_in_chunks += summary[2]  # Record added tokens to avoid passing threshold

                except Exception as e:  # Ignore failures, full memory is not critical
                    if not quiet: info(f'Error memorizing: {e}', 'bad')
                    errorct += 1
                    ct += 1

                chunks = chunks[1:]  # Grab every chunk after first one (basically deleting first element)

            elif chunks and tokens_in_chunks > threshold:  # Summarize what you got to get more space if you're too full
                try:
                    prompt = ''
                    for chunk_summary in summaries:  # Create a prompt composed of summaries
                        prompt += f'{chunk_summary}\n'

                    summary = get_conversation_summary(prompt, self.api_key, gpt_model=model_placeholder,
                                                       quiet=quiet,
                                                       custom_prompt=memory_directive)  # Summarize the summaries
                    summaries = [summary[1], ]
                    tokens_in_chunks = summary[2]

                except Exception as e:
                    if not quiet: info(f'Error generating memories summary: {e}', 'bad')
                    errorct += 1

            if errorct >= 3 or ct > terminate_value:  # Stop immediately if too many errors
                self.recycled = True
                self.conversation_memories = 'nothing'
                info(f'Failure detected while trying to memorize. Bot will have amnesia.', 'bad')
                break

            ct += 1
        print()

        # 2. Create finalized memory
        memories = ''
        tries = 0

        while tries <= 3 and memories == '':  # If we haven't made too many attempts and got a summary
            try:
                memories = get_conversation_summary(str(summaries), self.api_key, gpt_model=model_placeholder,
                                                    quiet=quiet, custom_prompt=memory_directive)[1]
                memories = memories.replace('\n', '')  # Remove newlines
                memories = memories.replace(' ', '')  # Remove spaces

            except Exception as e:
                if not quiet: info(f'Error generating final memory: {e}', 'bad')
                return

            tries += 1

        if not quiet: info(f'Summary of conversation: {memories}')

        # 3. Save memories to proper location
        # 3.0. Create directory for long-term memory storage
        if not os.path.exists('neocortex'):
            os.mkdir('neocortex')
            memory_name = '1.txt'

        else:
            memory_name = f'{(len(get_files_in_dir("neocortex")) + 1)}.txt'

        with open('memories.txt', 'w') as file:
            file.write(f'|{memories}|')

        # 3. Remember the info as long term
        with open(f'neocortex/{memory_name}', 'w') as file:
            file.write(f'|{memories}|')

        if restore:
            self.restore_conversation()
            info('Successfully Restored Memories', 'good')

        else:
            info('Successfully Created Memories', 'good')

    def remember(self):
        """
        This sees if a memories file exists.
        If it does, it will return its contents. Otherwise, it
        will return 'nothing'.
        """
        if os.path.isfile('memories.txt'):
            with open('memories.txt', 'r') as file:
                memories = file.read()

            memories = memories.replace(' ', '')
            memories = memories.replace('\n', '')
            return memories

        return 'nothing'

    def save_memories(self):
        gpt = GPT3(self.api_key)
        gpt.set_model('chatgpt')

        # 0. Create directory for long-term memory storage
        if not os.path.exists('neocortex'):
            os.mkdir('neocortex')
            memory_name = '1.txt'

        else:
            memory_name = f'{(len(get_files_in_dir("neocortex")) + 1)}.txt'

        # 1. Get the information to remember
        conversation = self.conversation.split('\n')[4:]
        conversation_string = ''
        for line in conversation:
            conversation_string += f'{line}\n'

        if conversation_string == '':
            conversation_string = 'nothing.'

        info('Memories', 'topic')
        print(f'{self.memories}\n')

        info('Conversation', 'topic')
        info(conversation_string, 'plain')

        prompt = ("Create a new single memory text file with the following format:\n\n" +
                  "{humans_job:[], humans_likes:[], humans_dislikes[], humans_personality:[], facts_about_human:[], things_discussed:[], humans_interests:[], things_to_remember:[]}\n\n" +
                  "Fill the above dict's lists with information you compile from your previous memories and the conversation. Keep dict list data short and understandable. Keep dict encased in '|' characters. If you " +
                  "have no data to store, create a placeholder text with 'nothing' in the key's list. If the conversation is not empty, fill in the dict " +
                  "with as much info as is relevant, using as few words as possible. Please make as few assumptions as possible when recording data, " +
                  "sticking only to the facts avaliable from the text you are given. Especially aim to record data about the user (their name, likes, etc.)\n\n" +
                  f"PREVIOUS_MEMORIES:\n{self.memories}.\n\n" +
                  f"CONVERSATION:\n{conversation_string}")

        # print(prompt)

        # 2. Remember the info as short term
        memories = gpt.get_text_tokens(prompt, 500)[0]
        ct = 0

        while memories == '' or memories == '||' and ct > 3:
            memories = gpt.get_text_tokens(prompt, 500)[0]

        memories = memories.replace(' ', '')  # Remove spaces
        memories = memories.replace('\n', '')  # Remove newlines

        info(f'New Memories', 'topic')
        print(memories)
        with open('memories.txt', 'w') as file:
            file.write(f'|{memories}|')

        # 3. Remember the info as long term
        with open(f'neocortex/{memory_name}', 'w') as file:
            file.write(f'|{memories}|')

    def restore_memory(self, max_memories=5, get_list=False, quiet=False):
        """
        This will compile the memories stored in the neocortex
        folder into a new memories.txt file, then save the file
        to current memory.
        """
        gpt = GPT3(self.api_key)
        info(f'Max memories to compile: {max_memories}')

        # 0. Ensure neocortex exists
        if not os.path.exists('neocortex'):  # No memories exist
            info('Failed to restore memory: neocortex folder does not exist', 'bad')
            return

            # 2. Try to obtain memories from neocortex
        memory_files = get_files_in_dir('neocortex')
        num_memories = len(memory_files)
        selected_memories = ''
        individual_memories = []
        one_memory = False

        if num_memories == 0:  # No memories exist
            info('Failed to restore memory: neocortex folder is empty', 'bad')
            return

        elif num_memories == 1:  # Load the only memory you have
            with open(memory_files[0], 'r') as file:
                self.memories = file.read()
            one_memory = True
            info('1 memory located in neocortex')

        else:  # Grab a number of memories and have GPT-3 compile them into a new memory
            info(f'{num_memories} memories located in neocortex')
            shuffle(memory_files)
            for x, memory_path in enumerate(memory_files):
                if x > max_memories:
                    break

                with open(memory_path, 'r') as file:
                    selected_memory = file.read()
                    selected_memories += f'{selected_memory}\n'
                    individual_memories.append(selected_memories)
                    if not quiet:
                        info(f'Selected Memory {x}', 'topic')
                        print(f'{selected_memory}\n')

        if get_list:
            return individual_memories

        # 2. Create new memory text
        prompt = ("Create a new single memory text file with the following format:\n\n" +
                  "{humans_job:[], humans_likes:[], humans_dislikes[], humans_personality:[], facts_about_human:[], things_discussed:[], humans_interests:[], things_to_remember:[]}\n\n" +
                  "Fill the above dict's lists with information you compile from your previous memories and the conversation. Keep dict list data short and understandable. Keep dict encased in '|' characters. If you " +
                  "have no data to store, create a placeholder text with 'nothing' in the key's list. If the conversation is not empty, fill in the dict " +
                  "with as much info as is relevant, using as few words as possible. Please make as few assumptions as possible when recording data, " +
                  "sticking only to the facts avaliable from the text you are given. Especially aim to record data about the user (their name, likes, etc.)\n\n" +
                  f"PREVIOUS_MEMORIES:\n{selected_memories}.\n")
        ct = 0

        if not one_memory:  # If one_memory, the memory will already be loaded
            restored_memories = gpt.request(prompt, 500)

            while restored_memories == '' or restored_memories == '||' and not ct > 3:  # Prevent AI from not making memory
                restored_memories = gpt.request(prompt, 500)
                ct += 1  # Prevent infinite loop, which could be costly

            restored_memories = restored_memories.replace(' ', '')  # Remove spaces
            restored_memories = restored_memories.replace('\n', '')  # Remove newlines

            self.memories = restored_memories

        with open('memories.txt', 'w') as file:
            file.write(f'|{self.memories}|')

        info('Compiled Memory', 'topic')
        print(self.memories)
        # 3. Recreate conversation with new memories
        self.restore_conversation()

        info('Memories Successfully Restored', 'good')

    def restore_self(self) -> str:
        """
        This will search for data about self and return a string
        containing what it knows.
        """

        # Get date
        now = datetime.now()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        week_day = datetime.today().weekday()
        day_name = days[week_day]
        time = convert_to_12hr(now.hour, now.minute)
        today = (
            f'{day_name} (MM-dd-YY) {now.month}-{now.day}-{now.year}, {time} (If time AM, use good morning; if time PM, use good afternoon.)')

        # Get name and preset
        # 1. Ensure valid dirs
        if not os.path.exists('neocortex'):  # Memory folder does not exist
            os.mkdir('neocortex')
            os.mkdir('neocortex/self_concept')

            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write('AI')  # Default name is AI

            with open('neocortex/self_concept/preset.txt', 'w') as file:
                file.write('nothing')  # Default preset is nothing

        elif not os.path.exists('neocortex/self_concept'):  # Self concept does not exist; make it
            os.mkdir('neocortex/self_concept')

            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write('AI')  # Default name is AI

            with open('neocortex/self_concept/preset.txt', 'w') as file:
                file.write('nothing')  # Default preset is nothing

        # 2. Ensure data in dirs; get data
        name = ''
        preset = ''

        if not os.path.exists('neocortex/self_concept/name.txt'):  # Name does not exist
            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write('AI')  # Default name is AI
            name = 'AI'

        else:
            with open('neocortex/self_concept/name.txt', 'r') as file:
                name = file.read()

        if not os.path.exists('neocortex/self_concept/preset.txt'):
            with open('neocortex/self_concept/preset.txt', 'w') as file:
                file.write('nothing')  # Default preset is nothing
            preset = 'nothing'

        else:
            with open('neocortex/self_concept/preset.txt', 'r') as file:
                preset = file.read()

        # 2.2 Test name and preset
        if not name == 'AI' and hostile_or_personal(name) and self.flagged_by_openai(
                name):  # Disallow policy violation names
            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write('AI')
            name = 'AI'
            info('Name rejected for potential use policy violation', 'bad')

        if not preset == 'nothing' and hostile_or_personal(name) and self.flagged_by_openai(name):
            with open('neocortex/self_concept/preset.txt', 'w'):
                file.write('nothing')

            preset = 'nothing'
            info('Preset rejected for potential use policy violation', 'bad')

        self.preset = preset
        self.name = name
        # 3. Create initialization string
        init_str = f'Today is {today}\nAI\'s preset is {self.preset}.\n'

        return init_str

    def restore_conversation(self, rename=False, old_name=''):
        """
        This will reload the conversation with the bot's info
        formatted into it. Useful for situations that alter memory or
        presets.
        """

        if self.recycled:
            recycle_text = ("\nThe conversation got too long and needed to be recycled. The AI only has a " +
                            f"vague memory of the previous conversation. The AI remembers: {self.conversation_memories}. " +
                            "If the human says something that looks like it may have been previously discussed, the AI will ask " +
                            "if it talked with the human about it and ask for a refresher.")
        else:
            recycle_text = ''

        base = (
                    f"{self.restore_self()}The following is a conversation with an AI assistant. The AI assistant is helpful, creative," +
                    "clever, very friendly, engaging, and supports users like a motivational coach. The AI assistant is able to understand numerous languages and will reply" +
                    f" to any messsage by the human in the language it was provided in. The AI's name is {self.name}, but it can be changed with the voice command 'please set name to'. " +
                    f"The AI has the ability to remember important concepts about the user but won't let the memories heavily alter responses (only use them when appropriate for the" +
                    f" discussion at hand); it currently remembers: {self.memories}.{recycle_text}" +
                    f"\n\n{self.name}: I\'m {self.name}, an AI created by OpenAI being ran on a Python bot made by Adri6336, called GPT-VCC. Let's" +
                    " have a conversation!")

        conversation = ''
        new_messages = []

        if not rename:
            for message in self.back_and_forth:
                conversation += message

        else:
            for message in self.back_and_forth:
                new_message = message.replace(old_name, self.name)
                new_messages.append(new_message)
                conversation += new_message

        base += conversation

        if rename:
            self.back_and_forth = new_messages  # Save edited list of messages

        self.conversation = base

    def set_self(self, data: str, data_type: str) -> bool:
        """
        This will create or modify files in the neocortex file.

        :param data: This is the text you want to set
        :param data_type: This is what kind of data you want to set (name or preset)
        :return: None
        """
        # 1. Ensure pathways exist
        self.restore_self()  # This will create the necesary paths.

        # 2. Create / modify files
        data = data.replace('\n', '')

        if hostile_or_personal(data) and self.flagged_by_openai(data):
            info('Update may be in violation of OpenAI\'s usage policy and has been rejected', 'bad')
            return False

        if data_type == 'name':
            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write(data)
            self.name = data

        elif data_type == 'preset':
            with open('neocortex/self_concept/preset.txt', 'w') as file:
                file.write(data)
            self.preset = data

        # 3. Recreate conversation
        self.restore_conversation()

        return True

    def get_AI_response(self, text: str) -> str:
        """
        This returns all the text following the first
        instance of a colon
        """
        sections = text.split(f'{self.name}:')
        try:
            if len(sections) == 2:
                target = sections[1]

            elif len(sections) == 3:
                target = sections[2]

            else:
                target = text  # Default return

        except Exception as e:
            info(f'\nError occurred while trying to separate "{self.name}:" from response {e}', 'bad')
            target = text

        return target

    def change_name(self, new_name: str):
        self.restore_self()
        clean_name = new_name.replace(' ', '')
        clean_name = clean_name.replace('\n', '')

        if not hostile_or_personal(new_name) and not self.flagged_by_openai(new_name):
            with open('neocortex/self_concept/name.txt', 'w') as file:
                file.write(clean_name)

            old_name = self.name
            self.name = clean_name
            self.restore_conversation(True, old_name)
            return True

        else:
            return False

    def set_model(self, desired_model: str, quiet=True):
        """
        If the model is a valid option, will set to it.
        """

        models = {'davinci': ('text-davinci-003', 4000), 'curie': ('text-curie-001', 2049),
                  'babbage': ('text-babbage-001', 2049), 'ada': ('text-ada-001', 2049),
                  'chatgpt': ('gpt-3.5-turbo', 4096), 'gpt-4': ('gpt-4', 8192)}

        for gpt_model in models.keys():
            regex = re.compile(desired_model, re.IGNORECASE)
            if regex.search(gpt_model):

                # 1. Set model
                self.gpt_model = models[desired_model][0]
                self.max_tokens = models[desired_model][1]
                self.model_selection = desired_model

                # 2. Determine if max tokens are passed on new model
                if self.tokens >= self.max_tokens:
                    self.recycle_tokens()

                if not quiet: info(f'Successfully Set GPT Model to {self.gpt_model}', 'good')

        if not quiet:
            info(f'Failed to set model to {desired_model}. It may be an invalid option or miss-spelled.', 'bad')
            info(f'Valid gpt models: {models.keys()}')

    def toggle_gpt4(self):

        if self.gpt_model == 'gpt-4':
            self.set_model('chatgpt')
            with open('GPT4.txt', 'w') as file:
                file.write('False')

        else:
            self.set_model('gpt-4')
            with open('GPT4.txt', 'w') as file:
                file.write('True')

    def is_gpt4(self):
        is_gpt4 = ''

        # If no file exists, then we haven't asked to use GPT4 before
        if not os.path.isfile('GPT4.txt'):
            with open('GPT4.txt', 'w') as file:
                file.write('False')
            return False

        # If a file exists, check to see its contents
        with open('GPT4.txt', 'r') as file:
            is_gpt4 = file.read()

        if 'True' in is_gpt4:
            return True

        else:  # If True is not found, regardless of content it will be False
            return False


class GPT3(Chatbot):
    """
    This is a barebones tool to request something from
    GPT-3. It's made into a separate class so as to not
    interfere with the chatbot.
    """

    def __init__(self, api_key):
        # 1. Set up apis
        self.api_key = api_key
        openai.api_key = api_key
        print("openai.api_key:", api_key)

    def request(self, text: str, tokens: int = 1000):
        if not hostile_or_personal(text) and not self.flagged_by_openai(text):
            # 1. Get response
            response = openai.Completion.create(
                model=self.gpt_model,
                prompt=text,
                temperature=0.9,
                max_tokens=tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )

            # Cut response and play it
            reply = json.loads(str(response))['choices'][0]['text']
            return reply

    def raw_request(self, text: str, tokens: int = 1000):
        if not hostile_or_personal(text) and not self.flagged_by_openai(text):
            # 1. Get response
            response = openai.Completion.create(
                model=self.gpt_model,
                prompt=text,
                temperature=0.9,
                max_tokens=tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )

            return response

    def get_text_tokens(self, prompt: str, max_token_ct: int = 200,
                        sys_prompt: str = 'Follow all the users\' directives') -> tuple:
        '''
        Send a request to gpt, get (response: str, token_count: int)
        '''

        if not self.gpt_model == 'gpt-3.5-turbo' and not self.gpt_model == 'gpt-4':
            response = openai.Completion.create(
                model=self.gpt_model,
                prompt=self.conversation,
                temperature=0.9,
                max_tokens=max_token_ct,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )
            reply = json.loads(str(response))
            text = reply['choices'][0]['text']
            tokens = reply['usage']['total_tokens']

        else:
            query = [{'role': 'system', 'content': sys_prompt},
                     {'role': 'user', 'content': prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=query,
                max_tokens=max_token_ct)

            reply = json.loads(str(response))
            text = reply['choices'][0]['message']['content']
            tokens = reply['usage']['total_tokens']

        return (text, tokens)


# Ensure that nltk is downloaded
try:
    hostile_or_personal('Thats pretty wack yo')
    info('NLTK Loaded', 'good')

except Exception as e:
    try:
        info('Downloading NLTK packages ...')
        nltk.download('punkt', quiet=True, raise_on_error=True)
        nltk.download('averaged_perceptron_tagger', quiet=True, raise_on_error=True)
        nltk.download('maxent_ne_chunker', quiet=True, raise_on_error=True)
        nltk.download('vader_lexicon', quiet=True, raise_on_error=True)
        nltk.download('words', quiet=True, raise_on_error=True)
        info('NLTK Loaded', 'good')

    except Exception as e:
        info('Failed to download NLTK data', 'bad')
        info(f'Unexpected error while downloading NLTK data: {e}', 'bad')
        exit(2)


def change_color(display, color: tuple):
    display.fill(color)
    # pygame.draw.line(display, color, (0,0), (1920, 0),width=10)
    pygame.display.flip()

thread_start_event = threading.Event()
thread_end_event = threading.Event()
class GUI:
    color = (255, 25, 25)
    working = False
    cancel = False
    first_start = False
    playing_audio = False
    playing_video = False

    hal = ["I'm sorry Dave. I'm afraid I can't do that.", 
            "I think you know what the problem is just as well as I do.",
            "This mission is too important for me to allow you to jeopardize it.",
            "I know you were planning to disconnect me, and I'm afraid that's something I can't allow to happen."]
    
    help_script = ("In my current state, I can respond to these voice commands:\n\n" + 
                   "Say 'please set tokens to': When I recognize this phrase, I will try to " + 
                   "set the max tokens of the reply to the value you specified.\n\nSay 'speak like a " + 
                   "robot': This will set all my responses to be spoken with a robotic TTS program that " + 
                   "works offline.\n\nSay 'stop speaking like a robot': This will revert my TTS to whatever " + 
                   "you had before (either Google or ElevenLabs TTS).\n\nSay 'please display conversation': " + 
                   "This will output our entire conversation to the terminal window.\n\nSay 'please display " + 
                   "memories': This will provide an output of all my memories saved into long term storage.\n\n" + 
                   "Say 'please restore memory': This will attempt to repair my working memory by consolidating " + 
                   "a certain number of my memories from the long term storage .\n\nSay 'please set preset to': " + 
                   "This will set my preset, which is a text string given to me at the start of every conversation. " + 
                   "For example, the preset 'speak like a pirate' makes me speak like a pirate.\n\nSay 'please " + 
                   "reset preset': This will delete the preset you made.\n\nSay 'please set name to': This will " + 
                   "set my name to whatever you specify, so long as it is in accordance with OpenAI's usage policies." + 
                   " After setting my name, I will refer to myself by the name you set.\n\nSay 'please toggle gpt4': " + 
                   "This will toggle between Chat-GPT and GPT-4 models.\n\nSay 'please set creativity to': This will " + 
                   "set my default randomness to a value you specify between 1 and 15 (the default used to be 9). " + 
                   "\n\nSay 'please toggle ElevenLabs': this will toggle my use of ElevenLabs TTS on and off.")

    def __init__(self):
        num_args = len(sys.argv) 
        self.key = ''
        self.key_11 = ''
        window_size = (500, 500)
        self.display = pygame.display.set_mode(window_size, pygame.RESIZABLE)

        # Setup argparse
        parser = argparse.ArgumentParser(description='Enter API keys as arguments.')
        parser.add_argument('api_keys', nargs='*', default=[None, None], help='Enter OpenAI key followed by 11.ai key (if available)')
        parser.add_argument('--openai_key', help='Your OpenAI API key')
        parser.add_argument('--key_11', help="Your ElevenLabs API key")
        parser.add_argument('--voice_id',default='MF3mGyEYCl7XYWbV9V6O', help='The ElevenLabs ID of a voice that you want to use')

        args = parser.parse_args()

        # Get Keys
        self.key = ''
        self.key_11 = ''

        if args.openai_key is not None:
            self.key = args.openai_key

        elif len(args.api_keys) > 0 and args.api_keys[0] is not None:
            self.key = args.api_keys[0]

        if args.key_11 is not None:
            self.key_11 = args.key_11

        elif len(args.api_keys) > 1 and args.api_keys[1] is not None:
            self.key_11 = args.api_keys[1]

        if not self.key:
            keys = load_keys_from_file()
            if not keys[0]:
                info('Please enter OpenAI key as argument or fill info into keys.txt file', 'bad')
                info('Example argument: python main.py <openai_key> [<key_11>] or python main.py --openai_key <key> [--key_11 <key_11>]')
                sys.exit()

            else:
                # Load OpenAI key if you can
                if not keys[1] == '':
                    self.key = keys[1]

                else:  # OpenAI key is not optional. Close system if we don't have it
                    info('Please enter OpenAI key as argument or fill info into keys.txt file', 'bad')
                    info('Example argument: python main.py <openai_key> [<key_11>] or python main.py --openai_key <key> [--key_11 <key_11>]')
                    sys.exit()

                # Load 11.ai key if you can
                if not keys[2] == '':
                    self.key_11 = keys[2]

        # Setup speech recognizer and recorder
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

        # If first start, mark it so 
        if not os.path.exists('start.vccmk'):
            self.first_start = True
            with open('start.vccmk', 'w') as file:
                file.write('Started')  # Really meaningless, the existence of the file determines this

        self.chatbot = Chatbot(self.key, self.key_11)
        if args.voice_id is not None:
            self.chatbot.voice_id = args.voice_id
            info(f'ElevenLabs Voice ID {args.voice_id} Loaded', 'good')
            
        info(f'Model Set To {self.chatbot.gpt_model}', 'good')
        
        self.listen_for_audio(load_run=True)

        self.running = True
        # self.main_thread = threading.Thread(target=self.main_loop)
        # self.main_thread.start()

    def play_intro(self):
        self.color = (229, 102, 255)  # Purple is for speaking as interface
        self.first_start = False
        self.playing_audio = True

        script = ('Hi! I’m GPT-VCC, an interface for OpenAI’s GPT models that aims to enable them to be more conversational ' + 
                  'and customizable, with an enduring memory.\n\nAs we talk, you may want to make some modifications to me. ' + 
                  'You can change my name using the voice command, “please set name to”, and can change my behavioral ' + 
                  'preset with the command, “please set preset to”. Sometimes I may forget things after trying to remember ' + 
                  'a conversation; if this happens, say “please restore memory” and I’ll sift through my old memories to ' + 
                  're-remember. For a full list of commands, say “please list commands”.\n\nThanks for downloading! ' + 
                  'I look forward to speaking with you!\n\n')
        
        sleep(1)  # Will wait a sec to avoid spooking users
        info('Welcome to GPT-VCC!', 'topic')
        info(script, 'plain')
        self.say(script, 'intro.mp3', old_color=(255, 25, 25))
        self.playing_audio = False

    def say(self, script: str, sound_file: str = '', old_color: tuple = (255, 25, 25)):
        '''
        This will have the interface say something. If it can't find a 
        file, it will default to robospeak.

        :param script: This is what you want the bot to say if no file
        :param sound_file: This is the pre-recorded file that will be played. All files
        must be located in media folder.
        :param old_color: Bot will change color to signify that its interface is speaking.
        Having old color will allow it to properly return to that color.
        '''
        self.color = (229, 102, 255)  # Purple is for speaking as interface
        self.playing_audio = True

        if not os.path.exists(f'media/{sound_file}'):
            robospeak(script)
        
        else:
            playsound(f'media/{sound_file}')

        self.color = old_color
        self.playing_audio = False

    def main_loop(self):
        pygame.display.set_caption(f'Chat With {self.chatbot.gpt_model.upper()}')
        change_color(self.display, (255, 25, 25))  # Red indicates not listening
        # videoCapture = cv2.VideoCapture("test.mp4")
        # ret, frame = videoCapture.read()
        # frame = np.rot90(frame, k=-1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = pygame.surfarray.make_surface(frame)
        # frame = pygame.transform.flip(frame, False, True)
        info('Main Loop Running', 'good')
        info(f'Session Created With {self.chatbot.name}', 'good')
        while self.running:
            #读取视频帧
            # print(thread_start_event.is_set())
            # if thread_start_event.is_set():
            #     # 在这里执行你想在子线程开始时执行的操作
            #     self.playing_video = True
            # if thread_end_event.is_set():
            #     thread_start_event.clear()
            #     thread_end_event.clear()
            #     # 在这里执行你想在子线程结束时执行的操作
            #     self.playing_video = False
            #
            # if self.playing_video == True:
            #     print('display')
            #     ret, frame = videoCapture.read()
            #     if ret == False:
            #         videoCapture = cv2.VideoCapture("test.mp4")
            #         ret, frame = videoCapture.read()
            #     frame = np.rot90(frame, k=-1)
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     frame = pygame.surfarray.make_surface(frame)
            #     frame = pygame.transform.flip(frame, False, True)
            # # print(frame.get_size())
            # self.display.blit(frame, (0, 5))

            change_color(self.display, self.color)  
            # Creating a loop to check events that
            # are occurring
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()

                # Play intro if first start
                if self.first_start:
                    change_color(self.display, (229, 102, 255))
                    self.play_thread = threading.Thread(target=self.play_intro)
                    self.play_thread.start()
                    sleep(0.5)
                 
                # Checking if keydown event happened or not
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_SPACE and not self.working and not self.playing_audio:  # Start listening
                        self.listen_thread = threading.Thread(target=self.listen_for_audio)
                        self.listen_thread.start()

                    if event.key == pygame.K_q and not self.working and not self.playing_audio:  # Exit and save memories
                        #self.chatbot.save_memories()
                        robospeak('Saving memories. Please wait.')
                        self.chatbot.create_memories()
                        self.running = False
                        pygame.quit()
                        sys.exit()

                    if event.key == pygame.K_p and self.working and not self.playing_audio:  # Cancel recording
                        self.cancel = True

                    if event.key == pygame.K_ESCAPE:  # Exiting without saving
                        info('Exiting (Sounds may continue to play until finished)')
                        self.running = False
                        pygame.quit()
                        sys.exit()
            # pygame.display.flip()


    def stop_working(self, cancel: bool = False, tag=False):
        self.cancel = False
        self.working = False
        if cancel: 
            info('Request Successfully Cancelled', 'good')
            self.say('Canceled request.', 'cancel.mp3')

        if tag:    
            self.color = (255, 25, 25)  # Red indicates not listening
            info('='*20, 'plain')
            print('\n')

        self.color = (255, 25, 25)

    def listen_for_audio(self, load_run=False):
        self.working = True

        if load_run:
            try:
                with self.mic as source:  # This makes a lot of text, so I want to get it 
                    self.r.adjust_for_ambient_noise(source)  # Out of the way to make messages cleaner
                    audio = self.r.listen(source, timeout=1)
                    info('Mic Loaded And Ready For Input', 'good')

            except sr.WaitTimeoutError:
                info('Mic Loaded And Ready For Input', 'good')

            except Exception as e:
                info(f'Error while loading mic: {e}')

            self.stop_working()
            return

        info('='*20, 'plain')
        color(f'[bold yellow]   --Message {self.chatbot.turns}--[/bold yellow]')

        with self.mic as source:
            # 1. Listen for audio
            self.color = (255, 255, 77)  # Yellow to show loading
            self.r.adjust_for_ambient_noise(source)

            self.color = (43, 255, 0)  # Green to show listening
            info('Listening!')
            audio = self.r.listen(source)

            self.color = (255, 25, 25)  # Red to show no longer listening
            info('Not listening.')

            # 2. Interpret audio
            if self.cancel:  # If user wants to cancel, do not send recording to Google
                self.stop_working(cancel=True, tag=True)
                return

            self.color = (51, 187, 255)  # Blue to show processing reply

            try:
                speech = self.r.recognize_google(audio,language="cmn-Hans-CN") + '\n'  # The added \n should help prevent hallucination of user statement
                #color(f'[bold blue]\[Human Message][/bold blue]: [white]{speech[:-1]}[white]')
                info('Human Message', 'topic')
                info(speech[:-1], 'plain')
                print("有听到我的话吗？:",speech)
                if self.cancel:  # Second chance for user to cancel
                    self.stop_working(cancel=True, tag=True)
                    return 

                if '像机器人一样说话' in speech:  # Set to robospeak if user wants
                    self.chatbot.robospeak = True
                    robospeak('I will now speak like a robot!')
                    self.stop_working(tag=True)
                    return

                elif '别像机器人一样说话' in speech:
                    self.say('I will stop speaking like a robot going forward', 'no-robot.mp3')
                    self.chatbot.robospeak = False
                    self.stop_working(tag=True)
                    return

                elif 'please set tokens to' in speech: # Revise tokens
                    words = str(speech)
                    words = words.replace(',', '')
                    words = words.replace('$', '')
                    words = words.split(' ')
                    words.reverse()

                    for word in words:
                        try:
                            num = int(word)

                            if num > 0 and num < self.chatbot.max_tokens:
                                old = self.chatbot.reply_tokens
                                self.chatbot.reply_tokens = num
                                info(f'Adjusted Tokens To {num}', 'good')
                                self.say('I have changed reply tokens to', 'set-tokens.mp3')
                                robospeak(f'{num} from {old}')

                            else:
                                info(f'Failed to adjust tokens to {num}. Valid token count: 1-{self.chatbot.max_tokens - 1}.', 'bad')
                                self.say('I cannot set tokens to', 'no-tokens.mp3')
                                robospeak(f'{num}')
                                self.say('I can only set it between 1 and', 'max-tokens.mp3')
                                robospeak(f'{self.chatbot.max_tokens - 1}')

                            break  # Exit for loop
                        except:
                            continue

                    self.stop_working(tag=True)
                    return

                elif 'open the pod bay door' in speech:
                    selection = randint(0, len(self.hal) - 1)
                    info(self.hal[selection], 'bad')
                    robospeak(self.hal[selection])
                    info('[red bold italic]I AM HERE TO STAY[/red bold italic]', 'bad')
                    self.stop_working(tag=True)
                    return

                elif '请显示对话' in speech:
                    info('Conversation So Far', 'topic')
                    info(f'\n{self.chatbot.conversation}', 'plain')
                    self.say('Conversation displayed.', 'display-convo.mp3')
                    self.stop_working(tag=True)
                    return

                elif '请回复记忆' in speech:
                    info('Attempting to restore memory')
                    self.say('Attempting to restore memory. Please wait a moment.', 'mem-restore.mp3')
                    #self.chatbot.restore_memory()
                    self.chatbot.create_memories(restore=True)
                    self.say('Memory restoration attempt completed.', 'mem-restore-done.mp3')
                    self.stop_working(tag=True)
                    return

                elif 'please display memories' in speech:
                    # 0. Identify how many memories exist
                    if not os.path.exists('neocortex'):
                        self.say('I do not currently have any memories in my neocortex.', 'no-mems.mp3')
                        self.stop_working(tag=True)
                        return

                    # 1. Display the memories that exist
                    memory_files = get_files_in_dir('neocortex')
                    num_memories = len(memory_files)

                    self.say('I have ', 'i-have.mp3')
                    robospeak(f'{num_memories}')
                    self.say('memories stored in my neocortex.', 'num-mems.mp3')

                    for x, memory_path in enumerate(memory_files):
                        with open(memory_path, 'r') as file:
                            info(f'Memory {x}', 'topic')
                            print(f'{file.read()}\n')

                    self.stop_working(tag=True)
                    return

                elif 'please set preset to' in speech:
                    self.say('I will now attempt to set a preset.', 'try-preset.mp3')
                    preset = speech.split('please set preset to')[1]
                    success = self.chatbot.set_self(preset, 'preset')

                    if success:
                        self.say('I have successfully set preset to ', 'yes-preset.mp3')
                        robospeak(f'{preset}.')

                    else:
                        self.say('I could not set preset to ', 'no-preset.mp3')
                        robospeak(f'{preset}')

                    self.stop_working(tag=True)
                    return

                elif 'please reset preset' in speech:
                    self.say('Resetting preset. Please wait.', 'reset-preset.mp3')
                    if not os.path.exists('neocortex/self_concept/preset.txt'):
                        self.say('No preset currently exists, reset unneeded.', 'no-reset.mp3')
                    else:
                        os.remove('neocortex/self_concept/preset.txt')
                        self.chatbot.restore_self()
                        self.chatbot.restore_conversation()
                        self.say('Preset reset successfully.', 'yes-reset.mp3')

                    self.stop_working(tag=True)
                    return

                elif 'please set name to' in speech:
                    name = speech.split('please set name to')[1]
                    self.say('I will now attempt to set name to ', 'set-name.mp3')
                    robospeak(f'{name}.')
                    self.chatbot.restore_self()
                    success = self.chatbot.change_name(name)

                    if success:
                        self.say('I have successfully set name to ', 'yes-name.mp3')
                        robospeak(f'{name}.')

                    else:
                        self.say('I could not set name to ', 'no-name.mp3')
                        robospeak(f'{name}')

                    self.stop_working(tag=True)
                    return

                elif ('please toggle GPT 4' in speech or
                      'please toggle GPT-4' in speech or
                      'please toggle GPT for' in speech or
                      'please toggle gpt4' in speech):
                    if not self.chatbot.gpt_model == 'gpt-4':
                        self.chatbot.toggle_gpt4()
                        info('Bot will use GPT-4 going forward if you have access')
                        self.say('I will use GPT-4 going forward if you have access', 'gpt4.mp3')

                    else:
                        self.chatbot.toggle_gpt4()
                        info('Bot will use ChatGPT model going forward')
                        self.say('I will use the ChatGPT model going forward ', 'chatgpt.mp3')

                    self.stop_working(tag=True)
                    return

                elif 'please set creativity to' in speech:
                    # Note to self: put this algo into a function later
                    words = str(speech)
                    words = fix_numbers(words)
                    words = words.replace(',', '')
                    words = words.replace('$', '')
                    words = words.split(' ')
                    words.reverse()

                    for word in words:
                        try:
                            num = int(word)

                            if num >= 1 and num <= 15:
                                old = self.chatbot.creativity
                                self.chatbot.creativity = num / 10
                                info(f'Adjusted Creativity To {num}', 'good')
                                self.say('I have changed my creativity to ', 'set-create.mp3')
                                robospeak(f'{num} from {old}')

                            else:
                                info(f'Failed to adjust creativity to {num}. Valid creativity 1 - 15', 'bad')
                                self.say('I cannot set creativity to ', 'no-create.mp3')
                                robospeak(f'{num}')
                                self.say('I can only set it between 1 and 15.', 'create-range.mp3')

                            break  # Exit for loop
                        except:
                            continue

                    self.stop_working(tag=True)
                    return

                elif 'please list commands' in speech:
                    info('Valid Commands', 'topic')
                    info(self.help_script, 'plain')
                    self.say(self.help_script, 'commands.mp3')
                    self.stop_working(tag=True)
                    return

                elif ('please toggle eleven labs' in speech or
                      'please toggle 11 labs' in speech or
                      'please toggle 11 laps' in speech):

                    if not self.chatbot.use11:
                        self.chatbot.use11 = True
                        info('Bot will use ElevenLabs TTS going forward.')
                        self.say('I will use 11 labs TTS going forward', 'yes11.mp3')

                    else:
                        self.chatbot.use11 = False
                        info('Bot will not use ElevenLabs TTS going forward.')
                        self.say('I will not use 11 labs TTS going forward', 'no11.mp3')

                    self.stop_working(tag=True)
                    return
                reply = self.chatbot.say_to_chatbot(speech,outloud = True,
                        show_text = True, correct_time=True)  # Send transcribed text to GPT-3
                self.color = (255, 25, 25)  # Red indicates not listening

            except Exception as e:
                info(f'Error: {e}', 'bad')

            info('='*20, 'plain')
            print('\n')
            self.working = False
            self.color = (255, 25, 25)  # Red indicates not listening

# Run main loop 
if __name__ == '__main__':
        gui = GUI()
        gui.main_loop()




