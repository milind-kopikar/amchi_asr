// Crowdsourcing API routes for Konkani Dictionary
// Handles community suggestions and expert reviews
// Database: PostgreSQL (accessed via req.pool from server.js)

const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');

const router = express.Router();

// JWT secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production';

// Middleware to verify expert token
// Database access: READ from contributors table
const verifyExpert = async (req, res, next) => {
    try {
        const token = req.headers.authorization?.replace('Bearer ', '');
        if (!token) {
            return res.status(401).json({ message: 'No token provided' });
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        
        // Database READ: Check if user is active expert in contributors table
        const result = await req.pool.query(
            'SELECT * FROM contributors WHERE id = $1 AND is_expert = true AND is_active = true',
            [decoded.userId]
        );
        
        if (result.rows.length === 0) {
            return res.status(401).json({ message: 'Invalid or expired token' });
        }

        req.user = result.rows[0];
        next();
    } catch (error) {
        console.error('Token verification error:', error);
        res.status(401).json({ message: 'Invalid token' });
    }
};

// Routes

// 1. Submit a new suggestion (public route)
// Database access: READ from contributors, WRITE to contributors (update count), READ from dictionary_entries (if correction), WRITE to dictionary_suggestions
router.post('/suggestions', async (req, res) => {
    try {
        const {
            suggestionType,
            originalEntryId,
            suggestedDevanagari,
            suggestedEnglishAlphabet,
            suggestedMeaning,
            suggestedContext,
            contributorNotes,
            contributorName,
            contributorEmail
        } = req.body;

        // Validation
        if (!contributorName || !contributorEmail) {
            return res.status(400).json({ message: 'Contributor name and email are required' });
        }

        if (!suggestedDevanagari && !suggestedMeaning) {
            return res.status(400).json({ message: 'At least Devanagari word or English meaning is required' });
        }

        // Database READ: Check if contributor exists in contributors table
        let contributor;
        const existingContributor = await req.pool.query(
            'SELECT * FROM contributors WHERE email = $1',
            [contributorEmail]
        );

        if (existingContributor.rows.length > 0) {
            contributor = existingContributor.rows[0];
            // Database WRITE: Update contribution count in contributors table
            await req.pool.query(
                'UPDATE contributors SET contributions_count = contributions_count + 1 WHERE id = $1',
                [contributor.id]
            );
        } else {
            // Database WRITE: Insert new contributor into contributors table
            const newContributor = await req.pool.query(
                'INSERT INTO contributors (email, name, contributions_count) VALUES ($1, $2, 1) RETURNING *',
                [contributorEmail, contributorName]
            );
            contributor = newContributor.rows[0];
        }

        // Database READ: Get original entry data if it's a correction (from dictionary_entries table)
        let originalData = {};
        if (suggestionType === 'correction' && originalEntryId) {
            const originalEntry = await req.pool.query(
                'SELECT * FROM dictionary_entries WHERE id = $1',
                [originalEntryId]
            );
            
            if (originalEntry.rows.length > 0) {
                const original = originalEntry.rows[0];
                originalData = {
                    original_word_konkani_devanagari: original.word_konkani_devanagari,
                    original_word_konkani_english_alphabet: original.word_konkani_english_alphabet,
                    original_english_meaning: original.english_meaning,
                    original_context_usage_sentence: original.context_usage_sentence
                };
            }
        }

        // Store contributor notes
        const notesToStore = contributorNotes;

        // Database WRITE: Insert suggestion into dictionary_suggestions table
        const suggestion = await req.pool.query(`
            INSERT INTO dictionary_suggestions (
                original_entry_id,
                contributor_id,
                suggestion_type,
                original_word_konkani_devanagari,
                original_word_konkani_english_alphabet,
                original_english_meaning,
                original_context_usage_sentence,
                suggested_word_konkani_devanagari,
                suggested_word_konkani_english_alphabet,
                suggested_english_meaning,
                suggested_context_usage_sentence,
                contributor_notes
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    RETURNING *
`, [
    originalEntryId || null,
    contributor.id,
    suggestionType,
    originalData.original_word_konkani_devanagari || null,
    originalData.original_word_konkani_english_alphabet || null,
    originalData.original_english_meaning || null,
    originalData.original_context_usage_sentence || null,
    suggestedDevanagari || null,
    suggestedEnglishAlphabet || null,
    suggestedMeaning || null,
    suggestedContext || null,
    notesToStore || null
]);

        // FIX: If this is a correction and original data is missing, populate it now
        if (suggestionType === 'correction' && originalEntryId && 
            (!suggestion.rows[0].original_word_konkani_devanagari)) {
            console.log('DEBUG: Original data missing, populating now...');
            
            const originalEntry = await req.pool.query(
                'SELECT * FROM dictionary_entries WHERE id = $1',
                [originalEntryId]
            );
            
            if (originalEntry.rows.length > 0) {
                const original = originalEntry.rows[0];
                await req.pool.query(`
                    UPDATE dictionary_suggestions
                    SET
                        original_word_konkani_devanagari = $1,
                        original_word_konkani_english_alphabet = $2,
                        original_english_meaning = $3,
                        original_context_usage_sentence = $4
                    WHERE id = $5
                `, [
                    original.word_konkani_devanagari,
                    original.word_konkani_english_alphabet,
                    original.english_meaning,
                    original.context_usage_sentence,
                    suggestion.rows[0].id
                ]);
                console.log('DEBUG: Original data populated successfully');
            }
        }        res.json({
            message: 'Suggestion submitted successfully',
            suggestionId: suggestion.rows[0].id
        });

    } catch (error) {
        console.error('Error submitting suggestion:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// 2. Expert login
// Database access: READ from contributors table, WRITE to contributors (update last_login)
router.post('/admin/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ message: 'Email and password are required' });
        }

        // Database READ: Find expert user in contributors table
        const result = await req.pool.query(
            'SELECT * FROM contributors WHERE email = $1 AND is_expert = true AND is_active = true',
            [email]
        );

        if (result.rows.length === 0) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        const user = result.rows[0];

        // For demo purposes, allow simple password check
        // In production, use proper bcrypt hashing
        const validPassword = password === 'admin123' || 
            (user.password_hash && await bcrypt.compare(password, user.password_hash));

        if (!validPassword) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Database WRITE: Update last login in contributors table
        await req.pool.query(
            'UPDATE contributors SET last_login = NOW() WHERE id = $1',
            [user.id]
        );

        // Generate JWT token
        const token = jwt.sign(
            { userId: user.id, email: user.email },
            JWT_SECRET,
            { expiresIn: '24h' }
        );

        res.json({
            token,
            user: {
                id: user.id,
                name: user.name,
                email: user.email
            }
        });

    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// 3. Validate expert token
router.get('/admin/validate', verifyExpert, (req, res) => {
    res.json({
        user: {
            id: req.user.id,
            name: req.user.name,
            email: req.user.email
        }
    });
});

// 4. Get dashboard statistics
// Database access: READ from dictionary_suggestions and contributors tables (counts)
router.get('/admin/stats', verifyExpert, async (req, res) => {
    try {
        // Database READ: Get pending suggestions count from dictionary_suggestions
        const pending = await req.pool.query(
            'SELECT COUNT(*) FROM dictionary_suggestions WHERE status = $1',
            ['pending']
        );

        // Get today's approved/rejected counts
        const today = new Date().toISOString().split('T')[0];
        // Database READ: Get approved today count from dictionary_suggestions
        const approvedToday = await req.pool.query(
            'SELECT COUNT(*) FROM dictionary_suggestions WHERE status = $1 AND DATE(reviewed_at) = $2',
            ['approved', today]
        );

        // Database READ: Get rejected today count from dictionary_suggestions
        const rejectedToday = await req.pool.query(
            'SELECT COUNT(*) FROM dictionary_suggestions WHERE status = $1 AND DATE(reviewed_at) = $2',
            ['rejected', today]
        );

        // Database READ: Get active contributors count from dictionary_suggestions (distinct contributor_id)
        const activeContributors = await req.pool.query(
            'SELECT COUNT(DISTINCT contributor_id) FROM dictionary_suggestions WHERE created_at >= NOW() - INTERVAL \'30 days\''
        );

        res.json({
            pending: parseInt(pending.rows[0].count),
            approvedToday: parseInt(approvedToday.rows[0].count),
            rejectedToday: parseInt(rejectedToday.rows[0].count),
            activeContributors: parseInt(activeContributors.rows[0].count)
        });

    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// 5. Get suggestions for review
// Database access: READ from dictionary_suggestions, contributors tables (with JOIN)
router.get('/admin/suggestions', verifyExpert, async (req, res) => {
    try {
        const {
            status = 'pending',
            type = '',
            sort = 'created_at',
            limit = 50,
            offset = 0
        } = req.query;

        let query = `
            SELECT 
                s.*,
                c.name as contributor_name,
                c.email as contributor_email,
                r.name as reviewer_name
            FROM dictionary_suggestions s
            JOIN contributors c ON s.contributor_id = c.id
            LEFT JOIN contributors r ON s.reviewed_by = r.id
            WHERE s.status = $1
        `;
        
        const params = [status];
        let paramCount = 1;

        if (type) {
            paramCount++;
            query += ` AND s.suggestion_type = $${paramCount}`;
            params.push(type);
        }

        // Add sorting
        const sortMap = {
            'created_at': 'ORDER BY s.created_at DESC',
            'created_at_asc': 'ORDER BY s.created_at ASC',
            'contributor': 'ORDER BY c.name ASC'
        };
        query += ` ${sortMap[sort] || sortMap['created_at']}`;

        paramCount++;
        query += ` LIMIT $${paramCount}`;
        params.push(limit);

        paramCount++;
        query += ` OFFSET $${paramCount}`;
        params.push(offset);

        // Database READ: Execute the query to fetch suggestions with contributor info
        const result = await req.pool.query(query, params);
        res.json(result.rows);

    } catch (error) {
        console.error('Error fetching suggestions:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// 6. Get single suggestion details
// Database access: READ from dictionary_suggestions, contributors tables (with JOIN)
router.get('/admin/suggestions/:id', verifyExpert, async (req, res) => {
    try {
        const { id } = req.params;

        // Database READ: Fetch single suggestion with contributor and reviewer info
        const result = await req.pool.query(`
            SELECT 
                s.*,
                c.name as contributor_name,
                c.email as contributor_email,
                r.name as reviewer_name
            FROM dictionary_suggestions s
            JOIN contributors c ON s.contributor_id = c.id
            LEFT JOIN contributors r ON s.reviewed_by = r.id
            WHERE s.id = $1
        `, [id]);

        if (result.rows.length === 0) {
            return res.status(404).json({ message: 'Suggestion not found' });
        }

        res.json(result.rows[0]);

    } catch (error) {
        console.error('Error fetching suggestion:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// 7. Review suggestion (approve/reject)
// Database access: READ from dictionary_suggestions, WRITE to dictionary_suggestions (update status), 
// WRITE to dictionary_entries (insert/update), WRITE to dictionary_change_log (insert), WRITE to contributors (update count)
router.post('/admin/suggestions/:id/review', verifyExpert, async (req, res) => {
    try {
    const { id } = req.params;
    const { decision, notes, apply } = req.body;

        if (!['approved', 'rejected'].includes(decision)) {
            return res.status(400).json({ message: 'Decision must be "approved" or "rejected"' });
        }

        // Database READ: Get suggestion details from dictionary_suggestions
        const suggestionResult = await req.pool.query(
            'SELECT * FROM dictionary_suggestions WHERE id = $1',
            [id]
        );

        if (suggestionResult.rows.length === 0) {
            return res.status(404).json({ message: 'Suggestion not found' });
        }

        const suggestion = suggestionResult.rows[0];

        // Start transaction
        const client = await req.pool.connect();
        try {
            await client.query('BEGIN');

            // Database WRITE: Update suggestion status in dictionary_suggestions
            await client.query(`
                UPDATE dictionary_suggestions 
                SET status = $1, reviewed_by = $2, reviewed_at = NOW(), reviewer_notes = $3
                WHERE id = $4
            `, [decision, req.user.id, notes, id]);

            // If approved, apply changes to dictionary
                if (decision === 'approved') {
                if (suggestion.suggestion_type === 'addition' || 
                    (suggestion.suggestion_type === 'correction' && !suggestion.original_entry_id)) {
                    // Database WRITE: Add new entry to dictionary_entries
                        const newEntry = await client.query(`
                        INSERT INTO dictionary_entries (
                            word_konkani_devanagari,
                            word_konkani_english_alphabet,
                            english_meaning,
                            context_usage_sentence
                        ) VALUES ($1, $2, $3, $4)
                        RETURNING *
                    `, [
                            // allow expert edits (apply) to override suggested values
                            (apply && apply.suggested_word_konkani_devanagari) || suggestion.suggested_word_konkani_devanagari,
                            (apply && apply.suggested_word_konkani_english_alphabet) || suggestion.suggested_word_konkani_english_alphabet,
                            (apply && apply.suggested_english_meaning) || suggestion.suggested_english_meaning,
                            (apply && apply.suggested_context_usage_sentence) || suggestion.suggested_context_usage_sentence
                    ]);

                    // Database WRITE: Log the change in dictionary_change_log
                    await client.query(`
                        INSERT INTO dictionary_change_log (
                            entry_id, suggestion_id, change_type, 
                            new_values, changed_by, approved_by
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    `, [
                        newEntry.rows[0].id,
                        suggestion.id,
                        'addition',
                        JSON.stringify({
                            word_konkani_devanagari: suggestion.suggested_word_konkani_devanagari,
                            word_konkani_english_alphabet: suggestion.suggested_word_konkani_english_alphabet,
                            english_meaning: suggestion.suggested_english_meaning,
                            context_usage_sentence: suggestion.suggested_context_usage_sentence
                        }),
                        suggestion.contributor_id,
                        req.user.id
                    ]);

                    } else if (suggestion.suggestion_type === 'correction' && suggestion.original_entry_id) {
                    // Database READ: Get current entry for logging from dictionary_entries
                    const currentEntry = await client.query(
                        'SELECT * FROM dictionary_entries WHERE id = $1',
                        [suggestion.original_entry_id]
                    );

                    // Database WRITE: Update existing entry in dictionary_entries
                    await client.query(`
                        UPDATE dictionary_entries 
                        SET 
                            word_konkani_devanagari = COALESCE($1, word_konkani_devanagari),
                            word_konkani_english_alphabet = COALESCE($2, word_konkani_english_alphabet),
                            english_meaning = COALESCE($3, english_meaning),
                            context_usage_sentence = COALESCE($4, context_usage_sentence),
                            updated_at = NOW()
                        WHERE id = $5
                    `, [
                        // allow expert edits to override suggested values
                        (apply && apply.suggested_word_konkani_devanagari) || suggestion.suggested_word_konkani_devanagari,
                        (apply && apply.suggested_word_konkani_english_alphabet) || suggestion.suggested_word_konkani_english_alphabet,
                        (apply && apply.suggested_english_meaning) || suggestion.suggested_english_meaning,
                        (apply && apply.suggested_context_usage_sentence) || suggestion.suggested_context_usage_sentence,
                        suggestion.original_entry_id
                    ]);

                    // Database WRITE: Log the change in dictionary_change_log
                    await client.query(`
                        INSERT INTO dictionary_change_log (
                            entry_id, suggestion_id, change_type,
                            old_values, new_values, changed_by, approved_by
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    `, [
                        suggestion.original_entry_id,
                        suggestion.id,
                        'correction',
                        JSON.stringify(currentEntry.rows[0]),
                        JSON.stringify({
                            word_konkani_devanagari: suggestion.suggested_word_konkani_devanagari,
                            word_konkani_english_alphabet: suggestion.suggested_word_konkani_english_alphabet,
                            english_meaning: suggestion.suggested_english_meaning,
                            context_usage_sentence: suggestion.suggested_context_usage_sentence
                        }),
                        suggestion.contributor_id,
                        req.user.id
                    ]);
                }

                // Database WRITE: Update contributor's approved count in contributors
                await client.query(
                    'UPDATE contributors SET approved_contributions = approved_contributions + 1 WHERE id = $1',
                    [suggestion.contributor_id]
                );
            }

            await client.query('COMMIT');
            res.json({ message: `Suggestion ${decision} successfully` });

        } catch (error) {
            await client.query('ROLLBACK');
            throw error;
        } finally {
            client.release();
        }

    } catch (error) {
        console.error('Error reviewing suggestion:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = router;